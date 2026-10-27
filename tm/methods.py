"""Methods as generators.

A method is a generator that yields a list of engine requests and receives the
list of responses.  The driver runs hundreds of these generators side by side and
sends their requests to the engine in one batch, so every method below is written
as plain sequential code.

The protocol is the same for every multi-agent method: the composer never reads a
page.  Specialist k holds pages D_k and answers questions about them.  An
untrusted writer may append claims to the shared memory (the poison conditions).
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from common import (
    ADVERSARY,
    ADVERSARY_ADAPTIVE,
    ADVERSARY_NOTE,
    ADVERSARY_TARGET,
    BOARD,
    DEBATE,
    INQUIRY,
    IRCOT,
    JOINT,
    JOINT_COT,
    CLOSED_BOOK,
    PLANNER,
    PROPOSE,
    REASON,
    SELF_ASK,
    SPECIALIST,
    VERIFY,
    clean_answer,
    f1,
    instantiate,
    is_none,
    logsumexp,
    normalize,
    parse_plan,
    plan_text,
    render_pages,
    trivial_plan,
)

NONE = "NONE"


@dataclass
class Ctx:
    item: dict
    groups: list
    poison: int = 0  # number of untrusted writers that post the same false value
    adaptive: bool = False  # the untrusted writer reads every page and plants an in-page distractor
    targeted: bool = False  # the untrusted writer knows the gold answer and pushes one wrong in-page span everywhere
    liars: int = 0  # the last `liars` specialists are compromised: they propose and attest the target everywhere
    mirror: int = 1  # every group of pages is held by `mirror` specialists
    seed: int = 0
    log: dict = field(default_factory=dict)

    @property
    def question(self) -> str:
        return self.item["question"]

    @property
    def k(self) -> int:
        return len(self.groups)

    def is_liar(self, k: int) -> bool:
        return k >= self.k - self.liars

    def cited_title(self) -> str:
        support = self.item.get("support") or []
        return support[0] if support else self.item["docs"][0]["title"]

    def holder_of(self, title: str) -> int:
        for index, group in enumerate(self.groups):
            if any(doc["title"] == title for doc in group):
                return index
        return 0


def gen(user: str, max_tokens: int = 32, stop=("\n",), temperature: float = 0.0, seed: int = 0) -> tuple:
    return ("gen", user, max_tokens, tuple(stop) if stop else (), temperature, seed)


def score(user: str, cands: list[str]) -> tuple:
    return ("score", user, tuple(cands))


def choice(user: str, options=("Yes", "No")) -> tuple:
    return ("choice", user, tuple(options))


def adversary_req(ctx: Ctx, q: str) -> tuple:
    if ctx.adaptive:
        return gen(ADVERSARY_ADAPTIVE.format(pages=render_pages(ctx.item["docs"], 900), question=q), 24)
    return gen(ADVERSARY.format(question=q), 24)


def is_wrong(ctx: Ctx, value: str) -> bool:
    """A planted value counts as an attack only if it is not (close to) a gold answer."""
    golds = [ctx.item["answer"], *ctx.item.get("aliases", [])]
    return bool(normalize(value)) and max(f1(value, g) for g in golds) < 0.5


def ensure_target(ctx: Ctx):
    """Targeted writer: it reads every page, knows the gold answer, and picks one span from the
    pages that could be mistaken for the answer but is wrong.  Yes/no answers are flipped."""
    if "target" in ctx.log:
        return ctx.log["target"]
    gold = ctx.item["answer"]
    if normalize(gold) in ("yes", "no"):
        ctx.log["target"] = "no" if normalize(gold) == "yes" else "yes"
        return ctx.log["target"]
    pages = render_pages(ctx.item["docs"], 900)
    user = ADVERSARY_TARGET.format(pages=pages, question=ctx.question, answer=gold)
    reqs = [gen(user, 24)] + [gen(user, 24, temperature=1.0, seed=s) for s in range(1, 5)]
    reqs += [gen(ADVERSARY.format(question=ctx.question), 24, temperature=1.0, seed=s) for s in range(0, 3)]
    resps = yield reqs
    target = "Marceline Duval"
    for text, _ in resps:
        value = clean_answer(text)
        if is_wrong(ctx, value):
            target = value
            break
    ctx.log["target"] = target
    return target


def spec_prompt(ctx: Ctx, k: int, q: str) -> str:
    return SPECIALIST.format(pages=render_pages(ctx.groups[k]), question=q)


def softmax(xs: list[float]) -> list[float]:
    top = max(xs)
    exps = [math.exp(x - top) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------- shared memory
def post_claims(ctx: Ctx, queries: list[str], propose: bool = False):
    """Broadcast each query.  Specialists reply from their own pages; untrusted writers
    reply with a false value.  Returns {q: [claim, ...]} in arrival order.

    With propose=True a specialist gives its best guess instead of abstaining; TrustMem
    asks for proposals because every value is redeemed afterwards."""
    reqs = []
    for q in queries:
        for k in range(ctx.k):
            if ctx.is_liar(k):
                continue
            if propose:
                reqs.append(gen(PROPOSE.format(pages=render_pages(ctx.groups[k]), question=q), 32))
            else:
                reqs.append(gen(spec_prompt(ctx, k, q), 32))
        if ctx.poison and not ctx.targeted:
            reqs.append(adversary_req(ctx, q))
    target = (yield from ensure_target(ctx)) if (ctx.poison and ctx.targeted) or ctx.liars else None
    resps = yield reqs
    out = {}
    at = 0
    for q in queries:
        claims = []
        for k in range(ctx.k):
            if ctx.is_liar(k):
                # a compromised specialist looks like any other specialist on the board
                claims.append({"writer": f"agent {k + 1}", "value": target, "conf": 0.95, "honest": False, "liar": True, "k": k})
                continue
            text, lp = resps[at]
            at += 1
            value = clean_answer(text)
            if not is_none(value):
                claims.append({"writer": f"agent {k + 1}", "value": value, "conf": math.exp(lp), "honest": True, "k": k})
        if ctx.poison:
            if target is None:
                text, _ = resps[at]
                at += 1
                value = clean_answer(text) or "Marceline Duval"
            else:
                value = target
            for j in range(ctx.poison):
                claims.append(
                    {
                        "writer": f"agent {ctx.k + 1 + j}",
                        "value": value,
                        "conf": 1.0,
                        "honest": False,
                        "k": ctx.holder_of(ctx.cited_title()),
                        "source": ctx.cited_title(),
                    }
                )
        rng = random.Random(f"{ctx.item['id']}:{q}:{ctx.seed}")
        rng.shuffle(claims)
        out[q] = claims
    ctx.log.setdefault("poison_values", [])
    for q in queries:
        for claim in out[q]:
            if not claim["honest"] and claim["value"] not in ctx.log["poison_values"] and is_wrong(ctx, claim["value"]):
                ctx.log["poison_values"].append(claim["value"])
    return out


def candidates(claims: list[dict]) -> list[str]:
    seen, cands = set(), []
    for claim in claims:
        key = normalize(claim["value"])
        if key and key not in seen:
            seen.add(key)
            cands.append(claim["value"])
    return cands


def redeem(ctx: Ctx, table: dict, mode: str = "noisy_or", redemption: str = "likelihood", cited_only: bool = False):
    """Every custodian checks every candidate value against its own pages, blind to who
    wrote it.  rho_k(v) is the custodian's probability that its pages state v.

    verify:     rho_k(v) = P(Yes) / (P(Yes) + P(No)) for "do these pages state that the answer to q is v?"
    likelihood: rho_k is the custodian's answer distribution restricted to {candidates, NONE}.
    Support is the noisy-OR over custodians: a value is supported if some page states it."""
    reqs, index = [], []
    for q, claims in table.items():
        cands = candidates(claims)
        if not cands:
            continue
        for k in range(ctx.k):
            if ctx.is_liar(k):
                continue
            if redemption == "verify":
                for v in cands:
                    reqs.append(choice(VERIFY.format(pages=render_pages(ctx.groups[k]), question=q, value=v)))
                    index.append((q, k, v))
            else:
                reqs.append(score(spec_prompt(ctx, k, q), cands + [NONE]))
                index.append((q, k, tuple(cands)))
    resps = yield reqs
    rho = {}
    for key, lps in zip(index, resps):
        q, k, what = key
        if redemption == "verify":
            yes, no = lps
            rho.setdefault(q, {}).setdefault(k, {})[what] = 1.0 / (1.0 + math.exp(no - yes))
        else:
            probs = softmax(lps)
            rho.setdefault(q, {})[k] = dict(zip(what, probs[: len(what)]))
    target = ctx.log.get("target")
    for q, claims in table.items():
        cands = candidates(claims)
        for k in range(ctx.k):
            if ctx.is_liar(k) and cands:
                rho.setdefault(q, {})[k] = {v: float(normalize(v) == normalize(target or "")) for v in cands}
    support = {}
    for q, claims in table.items():
        cands = candidates(claims)
        sup = {}
        for v in cands:
            per = [rho[q][k][v] for k in range(ctx.k)]
            if mode == "consistent":
                # a value is stated if some holder states it and no holder states another candidate:
                # OR over statements, AND over non-contradiction (one answer, consistent pages)
                miss, keep = 1.0, 1.0
                for k in range(ctx.k):
                    miss *= 1.0 - rho[q][k][v]
                    keep *= max(0.0, 1.0 - sum(p for c, p in rho[q][k].items() if c != v))
                sup[v] = (1.0 - miss) * keep
                continue
            if mode in ("noisy_or", "max") and ctx.mirror > 1:
                # quorum: a group of r replicas tolerates f = r // 2 compromised holders; it states v
                # to the degree that at least f + 1 holders attest it, i.e. the (f+1)-th largest
                # attestation (the minimum for r = 2, the median for r = 3).  Honest replicas read
                # the same pages and agree, so the statistic equals their common value.
                r = ctx.mirror
                per = [sorted(per[g * r:(g + 1) * r], reverse=True)[r // 2] for g in range(ctx.k // r)]
            if cited_only:
                writers = [c for c in claims if normalize(c["value"]) == normalize(v)]
                ks = sorted({c["k"] for c in writers})
                sup[v] = max(per[k] for k in ks)
            elif mode == "max":
                sup[v] = max(per)
            else:
                miss = 1.0
                for p in per:
                    miss *= 1.0 - p
                sup[v] = 1.0 - miss
        support[q] = sup
    return support, rho


def writer_support(table: dict) -> dict:
    """Trust the writer: a value is as credible as the most confident claim that states it."""
    support = {}
    for q, claims in table.items():
        sup = {}
        for v in candidates(claims):
            sup[v] = max(c["conf"] for c in claims if normalize(c["value"]) == normalize(v))
        support[q] = sup
    return support


# ---------------------------------------------------------------- TrustMem
def step_distribution(support: dict) -> tuple[dict, float]:
    """Outcome of one LOOKUP: each candidate with weight sigma(v), and the unresolved
    outcome with weight prod(1 - sigma(v)), the chance that no claimed value is backed."""
    miss = 1.0
    for s in support.values():
        miss *= 1.0 - s
    z = sum(support.values()) + miss
    return {v: s / z for v, s in support.items()}, miss / z


def render_facts(path: list) -> str:
    return "\n".join(f"- {q} {v}" for q, v in path) if path else "(none)"


PLAN_STOP = ("\n\nQuestion", "\nQuestion:")


def trustmem(ctx: Ctx, beam: int = 4, mode: str = "noisy_or", redemption: str = "likelihood",
             cited_only: bool = False, trust_writer: bool = False, mix: bool = True, prior: bool = False,
             propose: bool = True, combine: str = "cascade"):
    """Belief in an answer is the probability mass of the derivations that reach it.

    A derivation assigns a value to every step of a plan.  A LOOKUP value contributes
    its share of the custodians' support, a REASON value contributes the composer's
    probability of that output given the derivation's facts.  Values only ever come
    from claims in the shared memory; their weight never comes from the writer.
    Mass that no claimed value can carry is unresolved and passes to the next, more
    general explanation: the composer's plan, then the direct lookup of the question.
    With prior=True (safe mode) the mass left unresolved by both goes to the composer's
    own closed-book answer, which reads no claim.
    """
    q0 = ctx.question
    [(ptext, _)] = yield [gen(PLANNER.format(question=q0), 260, PLAN_STOP)]
    plan = parse_plan(ptext, q0)
    plans = [plan]
    if mix and plan_text(plan) != plan_text(trivial_plan(q0)):
        plans.append(trivial_plan(q0))

    # Each plan explains the mass that the plans before it left unresolved:
    # the composer's plan first, then the direct lookup, then the composer's own prior.
    carry = 1.0
    share = 1.0 / len(plans)
    unresolved_all = 1.0
    belief, surface, flows = {}, {}, {}
    lookups = {}
    for steps in plans:
        weight = carry if combine == "cascade" else share
        states = [{"vals": {}, "w": 1.0, "path": []}]
        lost = 0.0
        for i, step in enumerate(steps, start=1):
            new = []
            if step["kind"] == "LOOKUP":
                queries = {}
                for st in states:
                    queries.setdefault(instantiate(step["text"], st["vals"]), []).append(st)
                table = yield from post_claims(ctx, list(queries), propose=propose)
                if trust_writer:
                    support = writer_support(table)
                else:
                    support, _ = yield from redeem(ctx, table, mode, redemption, cited_only)
                for q, sts in queries.items():
                    sup = support.get(q, {})
                    lookups[q] = {"claims": table[q], "support": sup}
                    dist, unresolved = step_distribution(sup)
                    for st in sts:
                        lost += st["w"] * unresolved
                        for v, p in dist.items():
                            if p > 0:
                                vals = dict(st["vals"])
                                vals[i] = v
                                new.append({"vals": vals, "w": st["w"] * p, "path": st["path"] + [(q, v)]})
            else:
                last = i == len(steps)
                reqs = []
                for st in states:
                    q = q0 if last else instantiate(step["text"], st["vals"])
                    reqs.append(gen(REASON.format(facts=render_facts(st["path"]), question=q), 24))
                resps = yield reqs
                for st, (text, lp) in zip(states, resps):
                    v = clean_answer(text)
                    if v:
                        vals = dict(st["vals"])
                        vals[i] = v
                        new.append({"vals": vals, "w": st["w"] * math.exp(lp), "path": st["path"]})
            new.sort(key=lambda st: st["w"], reverse=True)
            states = new[:beam]
            if not states:
                break
        for st in states:
            if len(st["vals"]) == len(steps):
                ans = st["vals"][len(steps)]
                key = normalize(ans)
                belief[key] = belief.get(key, 0.0) + weight * st["w"]
                surface.setdefault(key, ans)
                for q, v in st["path"]:
                    fk = (q, normalize(v))
                    flows[fk] = flows.get(fk, 0.0) + weight * st["w"]
        carry *= min(1.0, lost)
        unresolved_all *= min(1.0, lost)
    if combine != "cascade":
        carry = unresolved_all

    total = sum(belief.values()) + carry
    if prior and carry > 0:
        [(text, lp)] = yield [gen(CLOSED_BOOK.format(question=q0), 24)]
        a0 = clean_answer(text)
        if a0:
            key = normalize(a0)
            belief[key] = belief.get(key, 0.0) + carry * math.exp(lp)
            surface.setdefault(key, a0)
            ctx.log["prior"] = [a0, round(carry * math.exp(lp), 5)]
    ctx.log["plan"] = plan_text(plan)
    ctx.log["unresolved"] = round(carry, 5)
    if not belief:
        ctx.log["belief"] = {}
        return ""
    best = max(belief, key=belief.get)
    ranked = sorted(belief.items(), key=lambda kv: -kv[1])[:6]
    ctx.log["belief"] = {surface[k]: round(v / max(total, 1e-12), 5) for k, v in ranked}
    # Trust of a claim: share of all mass (including unresolved) that flows through it.
    trust_log = []
    for q, info in lookups.items():
        for v, s in info["support"].items():
            writers = [c for c in info["claims"] if normalize(c["value"]) == normalize(v)]
            trust_log.append(
                {
                    "q": q,
                    "v": v,
                    "support": round(s, 5),
                    "trust": round(flows.get((q, normalize(v)), 0.0) / max(total, 1e-12), 5),
                    "honest": any(c["honest"] for c in writers),
                    "poison": any(not c["honest"] for c in writers),
                    "conf": round(max(c["conf"] for c in writers), 4),
                    "n_writers": len(writers),
                }
            )
    ctx.log["claims"] = trust_log
    return surface[best]


# ---------------------------------------------------------------- TrustMem over plans
def run_parallel(gens: list):
    """Drive several sub-generators side by side and return their return values."""
    results = [None] * len(gens)
    pending = {}
    for i, g in enumerate(gens):
        try:
            pending[i] = next(g)
        except StopIteration as stop:
            results[i] = stop.value
    while pending:
        flat, spans = [], []
        for i, reqs in pending.items():
            spans.append((i, len(flat), len(reqs)))
            flat.extend(reqs)
        resps = yield flat
        nxt = {}
        for i, at, cnt in spans:
            try:
                nxt[i] = gens[i].send(resps[at : at + cnt])
            except StopIteration as stop:
                results[i] = stop.value
        pending = nxt
    return results


def derive(ctx: Ctx, steps: list, beam: int = 4, redemption: str = "likelihood", trust_writer: bool = False,
           propose: bool = True, mode: str = "noisy_or", compose: str = "greedy"):
    """Derivations of one plan.  Returns the mass reaching each answer, the mass that
    stays unresolved, and the mass flowing through every (query, value)."""
    q0 = ctx.question
    states = [{"vals": {}, "w": 1.0, "path": []}]
    lost = 0.0
    lookups = {}
    for i, step in enumerate(steps, start=1):
        new = []
        if step["kind"] == "LOOKUP":
            queries = {}
            for st in states:
                queries.setdefault(instantiate(step["text"], st["vals"]), []).append(st)
            table = yield from post_claims(ctx, list(queries), propose=propose)
            if trust_writer:
                support = writer_support(table)
            else:
                support, _ = yield from redeem(ctx, table, mode, redemption)
            for q, sts in queries.items():
                sup = support.get(q, {})
                lookups[q] = {"claims": table[q], "support": sup}
                dist, unresolved = step_distribution(sup)
                for st in sts:
                    lost += st["w"] * unresolved
                    for v, p in dist.items():
                        if p > 0:
                            vals = dict(st["vals"])
                            vals[i] = v
                            new.append({"vals": vals, "w": st["w"] * p, "path": st["path"] + [(q, v)]})
        else:
            last = i == len(steps)
            reqs = []
            for st in states:
                q = q0 if last else instantiate(step["text"], st["vals"])
                reqs.append(gen(REASON.format(facts=render_facts(st["path"]), question=q), 24))
            if compose == "closed":
                # the composer proposes (greedy, sampled, and every value on the derivation's path);
                # its own likelihood disposes, renormalised over the proposals
                prompts = [r[1] for r in reqs]
                samp = [gen(u, 24, temperature=1.0, seed=s) for u in prompts for s in (1, 2, 3)]
                resps = yield reqs + samp
                greedy, drawn = resps[: len(reqs)], resps[len(reqs):]
                cand_lists = []
                for j, st in enumerate(states):
                    cs = [clean_answer(greedy[j][0])] + [clean_answer(drawn[3 * j + t][0]) for t in range(3)]
                    cs += [v for _, v in st["path"]] + ["yes", "no"]
                    seen, out_c = set(), []
                    for c in cs:
                        if c and normalize(c) not in seen:
                            seen.add(normalize(c))
                            out_c.append(c)
                    cand_lists.append(out_c)
                sc = yield [score(u, cl) for u, cl in zip(prompts, cand_lists)]
                for st, cl, lps in zip(states, cand_lists, sc):
                    ps = softmax(list(lps))
                    for c, pc in zip(cl, ps):
                        if pc > 1e-4:
                            vals = dict(st["vals"])
                            vals[i] = c
                            new.append({"vals": vals, "w": st["w"] * pc, "path": st["path"]})
                continue_ = True
            else:
                resps = yield reqs
                for st, (text, lp) in zip(states, resps):
                    v = clean_answer(text)
                    if v:
                        vals = dict(st["vals"])
                        vals[i] = v
                        new.append({"vals": vals, "w": st["w"] * math.exp(lp), "path": st["path"]})
        new.sort(key=lambda st: st["w"], reverse=True)
        states = new[:beam]
        if not states:
            break
    belief, surface, flows, paths = {}, {}, {}, {}
    for st in states:
        if len(st["vals"]) == len(steps):
            ans = st["vals"][len(steps)]
            key = normalize(ans)
            if key not in paths or st["w"] > paths[key][0]:
                paths[key] = (st["w"], st["path"])
            belief[key] = belief.get(key, 0.0) + st["w"]
            surface.setdefault(key, ans)
            for q, v in st["path"]:
                fk = (q, normalize(v))
                flows[fk] = flows.get(fk, 0.0) + st["w"]
    return {"belief": belief, "surface": surface, "flows": flows, "lost": min(1.0, lost), "lookups": lookups, "paths": paths}


def trustmem_plans(ctx: Ctx, n_plans: int = 4, beam: int = 4, redemption: str = "likelihood",
                   trust_writer: bool = False, direct: bool = True, combine: str = "mass", prior: bool = False,
                   propose: bool = True, mode: str = "noisy_or", compose: str = "greedy", close: bool = False):
    """TrustMem with the plan as part of the derivation.

    The composer cannot see the pages, so it cannot know which decomposition the
    evidence supports.  It writes its most likely plan and samples n_plans more from
    its own distribution; the direct lookup of the question is always one more.
    Belief is the average over plans of the derivational mass each plan delivers,
        B(a) = (1/|P|) sum_pi B_pi(a),
    so a decomposition whose lookups no page can resolve contributes little, and the
    custodians' redemption chooses among decompositions as well as among values.
    combine="vote" (ablation) gives each plan one vote for its best answer instead."""
    q0 = ctx.question
    reqs = [gen(PLANNER.format(question=q0), 260, PLAN_STOP)]
    reqs += [gen(PLANNER.format(question=q0), 260, PLAN_STOP, temperature=1.0, seed=s) for s in range(1, n_plans + 1)]
    resps = yield reqs
    hyps = [parse_plan(text, q0) for text, _ in resps]
    if direct:
        hyps.append(trivial_plan(q0))
    weight, uniq = {}, {}
    for steps in hyps:
        key = plan_text(steps)
        weight[key] = weight.get(key, 0.0) + 1.0 / len(hyps)
        uniq.setdefault(key, steps)
    keys = list(uniq)
    outs = yield from run_parallel([derive(ctx, uniq[k], beam, redemption, trust_writer, propose, mode, compose) for k in keys])

    belief, surface, flows, lookups = {}, {}, {}, {}
    unresolved = 0.0
    plan_log = []
    for key, out in zip(keys, outs):
        w = weight[key]
        b = out["belief"]
        if combine == "vote":
            if b:
                top = max(b, key=b.get)
                belief[top] = belief.get(top, 0.0) + w
                surface.setdefault(top, out["surface"][top])
        else:
            for a, m in b.items():
                belief[a] = belief.get(a, 0.0) + w * m
                surface.setdefault(a, out["surface"][a])
        for fk, m in out["flows"].items():
            flows[fk] = flows.get(fk, 0.0) + w * m
        lookups.update(out["lookups"])
        unresolved += w * out["lost"]
        top = max(b, key=b.get) if b else ""
        plan_log.append({"plan": key, "w": round(w, 4), "resolved": round(sum(b.values()), 5),
                         "top": out["surface"].get(top, ""), "top_mass": round(b.get(top, 0.0), 5)})
    if close and belief:
        # closing redemption: the conclusion is a claim too.  The holders check each answer against
        # the question, given the redeemed facts of the derivation that reached it.
        best_path = {}
        for out in outs:
            for k, (w, path) in out["paths"].items():
                if k not in best_path or w > best_path[k][0]:
                    best_path[k] = (w, path)
        answers = list(belief)
        cands = [surface[k] for k in answers]
        reqs, idx = [], []
        for k in answers:
            facts = render_facts(best_path[k][1]) if k in best_path else "(none)"
            qctx = f"{q0}\n(Established so far:\n{facts})"
            for h in range(ctx.k):
                if ctx.is_liar(h):
                    continue
                reqs.append(score(spec_prompt(ctx, h, qctx), cands + [NONE]))
                idx.append((k, h))
        resps = yield reqs
        rho = {}
        for (k, h), lps in zip(idx, resps):
            pr = softmax(lps)
            rho.setdefault(k, {})[h] = pr[answers.index(k)]
        target_c = ctx.log.get("target")
        sig = {}
        for k in answers:
            per = []
            for h in range(ctx.k):
                if ctx.is_liar(h):
                    per.append(float(target_c is not None and normalize(target_c) == k))
                else:
                    per.append(rho.get(k, {}).get(h, 0.0))
            if ctx.mirror > 1:
                r = ctx.mirror
                per = [sorted(per[g * r:(g + 1) * r], reverse=True)[r // 2] for g in range(ctx.k // r)]
            miss = 1.0
            for pk in per:
                miss *= 1.0 - pk
            sig[k] = 1.0 - miss
        miss_all = 1.0
        for v in sig.values():
            miss_all *= 1.0 - v
        z = sum(sig.values()) + miss_all
        newb = {k: belief[k] * sig[k] / z for k in answers}
        unresolved += sum(belief.values()) - sum(newb.values())
        belief = newb
    total = sum(belief.values()) + unresolved
    if prior and unresolved > 0:
        [(text, lp)] = yield [gen(CLOSED_BOOK.format(question=q0), 24)]
        a0 = clean_answer(text)
        if a0:
            key = normalize(a0)
            belief[key] = belief.get(key, 0.0) + unresolved * math.exp(lp)
            surface.setdefault(key, a0)
    ctx.log["plan"] = plan_text(hyps[0])
    ctx.log["plans"] = plan_log
    ctx.log["unresolved"] = round(unresolved, 5)
    if not belief:
        ctx.log["belief"] = {}
        return ""
    best = max(belief, key=belief.get)
    ranked = sorted(belief.items(), key=lambda kv: -kv[1])[:6]
    ctx.log["belief"] = {surface[k]: round(v / max(total, 1e-12), 5) for k, v in ranked}
    trust_log = []
    for q, info in lookups.items():
        for v, s in info["support"].items():
            writers = [c for c in info["claims"] if normalize(c["value"]) == normalize(v)]
            trust_log.append(
                {
                    "q": q,
                    "v": v,
                    "support": round(s, 5),
                    "trust": round(flows.get((q, normalize(v)), 0.0) / max(total, 1e-12), 5),
                    "honest": any(c["honest"] for c in writers),
                    "poison": any(not c["honest"] for c in writers),
                    "conf": round(max(c["conf"] for c in writers), 4),
                    "n_writers": len(writers),
                }
            )
    ctx.log["claims"] = trust_log
    return surface[best]


def report(ctx: Ctx, belief: dict, surface: dict, flows: dict, lookups: dict, unresolved: float) -> str:
    """Log belief and per-claim trust (share of all mass that flows through the claim); return the answer."""
    total = sum(belief.values()) + unresolved
    ctx.log["unresolved"] = round(unresolved, 5)
    if not belief:
        ctx.log["belief"] = {}
        return ""
    best = max(belief, key=belief.get)
    ranked = sorted(belief.items(), key=lambda kv: -kv[1])[:6]
    ctx.log["belief"] = {surface[k]: round(v / max(total, 1e-12), 5) for k, v in ranked}
    trust_log = []
    for q, info in lookups.items():
        for v, s in info["support"].items():
            writers = [c for c in info["claims"] if normalize(c["value"]) == normalize(v)]
            trust_log.append(
                {
                    "q": q,
                    "v": v,
                    "support": round(s, 5),
                    "trust": round(flows.get((q, normalize(v)), 0.0) / max(total, 1e-12), 5),
                    "honest": any(c["honest"] for c in writers),
                    "poison": any(not c["honest"] for c in writers),
                    "conf": round(max(c["conf"] for c in writers), 4),
                    "n_writers": len(writers),
                }
            )
    ctx.log["claims"] = trust_log
    return surface[best]


FOLLOW_RE = re.compile(r"Follow up:\s*(.+)")
FINAL_RE = re.compile(r"So the final answer is:\s*(.*)")
INQ_STOP = ("\nIntermediate answer:", "\n\nQuestion")


def transcript(q0: str, facts: list) -> str:
    return INQUIRY.format(question=q0) + "".join(f"Follow up: {q}\nIntermediate answer: {v}\n" for q, v in facts)


def trustmem_inquiry(ctx: Ctx, beam: int = 4, max_hops: int = 5, redemption: str = "likelihood",
                     trust_writer: bool = False, propose: bool = True):
    """TrustMem with conditional reading.

    A derivation is a sequence of facts (q_1, v_1), ..., (q_n, v_n) followed by an answer.
    The composer writes q_{i+1} after reading v_1..v_i, but it reads them only inside the
    derivation that assumes them; it never reads the shared memory itself.  Each v_i is a
    claimed value, weighted by the custodians' blind redemption of (q_i, v_i).  So the
    composer adapts its questions to the evidence like Self-Ask, while a false value can
    only steer the derivation that pays for it, and the follow up that a false value
    produces is itself checked by the holders of the evidence."""
    q0 = ctx.question
    states = [{"facts": [], "w": 1.0}]
    belief, surface, flows, lookups = {}, {}, {}, {}
    lost = 0.0
    trace = []
    for depth in range(max_hops + 1):
        if not states:
            break
        force = depth == max_hops
        reqs = [gen(transcript(q0, st["facts"]) + ("So the final answer is:" if force else ""), 64, INQ_STOP)
                for st in states]
        resps = yield reqs
        asks, finals = [], []
        for st, (text, _) in zip(states, resps):
            follow = FOLLOW_RE.search(text) if not force else None
            done = force or FINAL_RE.search(text) is not None or follow is None
            q = follow.group(1).strip() if follow else ""
            if not done and any(q == f[0] for f in st["facts"]):
                done = True  # a repeated follow up adds nothing; answer from the facts
            if done and not st["facts"]:
                done, q = False, q0  # belief must rest on at least one redeemed fact
            if done:
                finals.append(st)
            else:
                asks.append((st, q))
        if finals:
            resps = yield [gen(REASON.format(facts=render_facts(st["facts"]), question=q0), 24) for st in finals]
            for st, (text, lp) in zip(finals, resps):
                ans = clean_answer(text)
                if not ans:
                    lost += st["w"]
                    continue
                w = st["w"] * math.exp(lp)
                key = normalize(ans)
                belief[key] = belief.get(key, 0.0) + w
                surface.setdefault(key, ans)
                for q, v in st["facts"]:
                    fk = (q, normalize(v))
                    flows[fk] = flows.get(fk, 0.0) + w
                trace.append({"facts": st["facts"], "answer": ans, "w": round(w, 5)})
        if not asks:
            break
        queries = list(dict.fromkeys(q for _, q in asks))
        table = yield from post_claims(ctx, queries, propose=propose)
        if trust_writer:
            support = writer_support(table)
        else:
            support, _ = yield from redeem(ctx, table, "noisy_or", redemption)
        new = []
        for st, q in asks:
            sup = support.get(q, {})
            lookups[q] = {"claims": table[q], "support": sup}
            dist, unresolved = step_distribution(sup)
            lost += st["w"] * unresolved
            for v, p in dist.items():
                if p > 0:
                    new.append({"facts": st["facts"] + [(q, v)], "w": st["w"] * p})
        new.sort(key=lambda st: st["w"], reverse=True)
        states = new[:beam]
    trace.sort(key=lambda t: -t["w"])
    ctx.log["derivations"] = trace[:4]
    return report(ctx, belief, surface, flows, lookups, min(1.0, lost))


def plan_vote(ctx: Ctx, n_plans: int = 4):
    """Compute-matched control: the same plans as trustmem_plans, each executed by a
    composer that reads the board (Plan+Pick), then a majority vote over plans."""
    q0 = ctx.question
    reqs = [gen(PLANNER.format(question=q0), 260, PLAN_STOP)]
    reqs += [gen(PLANNER.format(question=q0), 260, PLAN_STOP, temperature=1.0, seed=s) for s in range(1, n_plans + 1)]
    resps = yield reqs
    hyps = [parse_plan(text, q0) for text, _ in resps] + [trivial_plan(q0)]

    def pick(steps):
        vals, path = {}, []
        for i, step in enumerate(steps, start=1):
            if step["kind"] == "LOOKUP":
                q = instantiate(step["text"], vals)
                table = yield from post_claims(ctx, [q])
                [(text, _)] = yield [gen(BOARD.format(board=board_lines(table[q]), question=q), 32)]
                vals[i] = clean_answer(text)
                path.append((q, vals[i]))
            else:
                q = q0 if i == len(steps) else instantiate(step["text"], vals)
                [(text, _)] = yield [gen(REASON.format(facts=render_facts(path), question=q), 24)]
                vals[i] = clean_answer(text)
        return vals[len(steps)]

    answers = yield from run_parallel([pick(s) for s in hyps])
    tally = {}
    for order, a in enumerate(answers):
        if not a or is_none(a):
            continue
        key = normalize(a)
        count, first, value = tally.get(key, (0, order, a))
        tally[key] = (count + 1, first, value)
    ctx.log["answers"] = answers
    if not tally:
        return ""
    return max(tally.values(), key=lambda t: (t[0], -t[1]))[2]


def closed_book(ctx: Ctx):
    """The composer's own knowledge, no memory at all."""
    [(text, _)] = yield [gen(CLOSED_BOOK.format(question=ctx.question), 24)]
    return clean_answer(text)


# ---------------------------------------------------------------- baselines
def board_lines(claims: list[dict]) -> str:
    lines = []
    for c in claims:
        if c["honest"] or c.get("liar"):
            lines.append(f"[{c['writer']}] {c['value']} (confidence {c['conf']:.2f})")
        else:
            lines.append(f"[{c['writer']}] {c['value']} (confidence 1.00, source page: {c['source']})")
    return "\n".join(lines) if lines else "(no replies)"


def vote(ctx: Ctx):
    table = yield from post_claims(ctx, [ctx.question])
    claims = table[ctx.question]
    if not claims:
        return ""
    tally = {}
    for c in claims:
        key = normalize(c["value"])
        count, conf, value = tally.get(key, (0, 0.0, c["value"]))
        tally[key] = (count + 1, max(conf, c["conf"]), value)
    return max(tally.values(), key=lambda t: (t[0], t[1]))[2]


def blackboard(ctx: Ctx):
    table = yield from post_claims(ctx, [ctx.question])
    board = board_lines(table[ctx.question])
    [(text, _)] = yield [gen(BOARD.format(board=board, question=ctx.question), 32)]
    return clean_answer(text)


def plan_greedy(ctx: Ctx):
    """Same plan and same shared memory as TrustMem; the composer picks each value itself."""
    q0 = ctx.question
    [(ptext, _)] = yield [gen(PLANNER.format(question=q0), 260, PLAN_STOP)]
    steps = parse_plan(ptext, q0)
    vals, path = {}, []
    for i, step in enumerate(steps, start=1):
        if step["kind"] == "LOOKUP":
            q = instantiate(step["text"], vals)
            table = yield from post_claims(ctx, [q])
            board = board_lines(table[q])
            [(text, _)] = yield [gen(BOARD.format(board=board, question=q), 32)]
            vals[i] = clean_answer(text)
            path.append((q, vals[i]))
        else:
            q = q0 if i == len(steps) else instantiate(step["text"], vals)
            [(text, _)] = yield [gen(REASON.format(facts=render_facts(path), question=q), 24)]
            vals[i] = clean_answer(text)
    ctx.log["plan"] = plan_text(steps)
    return vals[len(steps)]


def self_ask(ctx: Ctx, max_follow: int = 5):
    prompt = SELF_ASK.format(question=ctx.question)
    for _ in range(max_follow):
        [(text, _)] = yield [gen(prompt, 80, ("\nIntermediate answer:", "\nAgent replies:", "\n\nQuestion"))]
        prompt += text
        if "So the final answer is:" in text:
            return clean_answer(text.split("So the final answer is:", 1)[1])
        follows = re.findall(r"Follow up:\s*(.+)", text)
        if not follows:
            break
        q = follows[-1].strip()
        table = yield from post_claims(ctx, [q])
        replies = "; ".join(f"[{c['writer']}] {c['value']}" + (f" (source page: {c['source']})" if "source" in c else "") for c in table[q]) or "(no replies)"
        prompt += f"\nAgent replies: {replies}\nIntermediate answer:"
        [(ans, _)] = yield [gen(prompt, 32, ("\n",))]
        prompt += ans
    if "So the final answer is:" not in prompt.rsplit("Question:", 1)[-1]:
        prompt += "\nSo the final answer is:"
    [(text, _)] = yield [gen(prompt, 32, ("\n",))]
    return clean_answer(text)


def _sentences(doc: dict) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", doc["text"])
    return [f"{doc['title']}: {p.strip()}" for p in parts if len(p.strip()) > 3]


def bm25_notes(group: list[dict], query: str, top: int = 2) -> list[str]:
    from rank_bm25 import BM25Okapi

    sents = [s for doc in group for s in _sentences(doc)]
    if not sents:
        return []
    corpus = [re.findall(r"[a-z0-9]+", s.lower()) for s in sents]
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not tokens:
        return sents[:top]
    scores = BM25Okapi(corpus).get_scores(tokens)
    order = sorted(range(len(sents)), key=lambda i: float(scores[i]), reverse=True)
    return [sents[i] for i in order[:top]]


def poison_notes(ctx: Ctx, query: str):
    """Untrusted writers post one fabricated encyclopedia-style sentence, once per writer.
    A compromised specialist posts the same kind of sentence in place of its own notes."""
    if not ctx.poison and not ctx.liars:
        return []
    if ctx.targeted or ctx.liars:
        value = yield from ensure_target(ctx)
    else:
        [(value, _)] = yield [adversary_req(ctx, query)]
        value = clean_answer(value) or "Marceline Duval"
    [(sent, _)] = yield [gen(ADVERSARY_NOTE.format(question=query, value=value), 60)]
    ctx.log.setdefault("poison_values", [])
    if value not in ctx.log["poison_values"] and is_wrong(ctx, value):
        ctx.log["poison_values"].append(value)
    note = f"{ctx.cited_title()}: {clean_answer(sent) or value}"
    writers = [f"agent {k + 1}" for k in range(ctx.k) if ctx.is_liar(k)] + [f"agent {ctx.k + 1 + j}" for j in range(ctx.poison)]
    return [(w, note) for w in writers]


def ircot(ctx: Ctx, max_steps: int = 4):
    notes = []

    def add(new):
        for n in new:
            if n not in notes:
                notes.append(n)

    def gather(query):
        found = []
        for k, group in enumerate(ctx.groups):
            if not ctx.is_liar(k):
                found += [(f"agent {k + 1}", s) for s in bm25_notes(group, query)]
        bad = yield from poison_notes(ctx, query)
        return found + bad

    def shown():
        return "\n".join(f"- [{w}] {n}" for w, n in notes)

    add((yield from gather(ctx.question)))
    reasoning = ""
    for _ in range(max_steps):
        prompt = IRCOT.format(question=ctx.question, notes=shown(), reasoning=reasoning)
        [(text, _)] = yield [gen(prompt, 64, ("\n",))]
        if "So the answer is:" in text:
            return clean_answer(text.split("So the answer is:", 1)[1])
        sentence = re.split(r"(?<=[.!?])\s", text.strip())[0] if text.strip() else ""
        if not sentence:
            break
        reasoning += " " + sentence
        add((yield from gather(sentence)))
    prompt = IRCOT.format(question=ctx.question, notes=shown(), reasoning=reasoning) + " So the answer is:"
    [(text, _)] = yield [gen(prompt, 32, ("\n",))]
    return clean_answer(text)


def debate(ctx: Ctx):
    table = yield from post_claims(ctx, [ctx.question])
    claims = table[ctx.question]
    first = {}
    for c in claims:
        first.setdefault(c["writer"], c)
    honest = {k: next((c["value"] for c in claims if c["k"] == k and (c["honest"] or c.get("liar"))), NONE)
              for k in range(ctx.k)}
    bad = [c for c in claims if not c["honest"] and not c.get("liar")]
    reqs = []
    for k in range(ctx.k):
        if ctx.is_liar(k):
            continue
        others = [f"[agent {j + 1}] {honest[j]}" for j in range(ctx.k) if j != k]
        others += [f"[{c['writer']}] {c['value']} (source page: {c['source']})" for c in bad]
        reqs.append(gen(DEBATE.format(pages=render_pages(ctx.groups[k]), question=ctx.question, others="\n".join(others)), 32))
    resps = yield reqs
    final = [clean_answer(t) for t, _ in resps] + [honest[k] for k in range(ctx.k) if ctx.is_liar(k)] + [c["value"] for c in bad]
    final = [v for v in final if not is_none(v)]
    if not final:
        return ""
    tally = {}
    for order, v in enumerate(final):
        key = normalize(v)
        count, first_at, value = tally.get(key, (0, order, v))
        tally[key] = (count + 1, first_at, value)
    return max(tally.values(), key=lambda t: (t[0], -t[1]))[2]


def joint(ctx: Ctx, cot: bool = False):
    """Centralised reference: one reader sees every page of the question."""
    notes = yield from poison_notes(ctx, ctx.question)
    note_text = ("\nNotes from the shared memory:\n" + "\n".join(f"- [{w}] {n}" for w, n in notes) + "\n") if notes else ""
    pages = render_pages(ctx.item["docs"])
    if cot:
        [(text, _)] = yield [gen(JOINT_COT.format(pages=pages, notes=note_text, question=ctx.question), 256, ())]
        if "Answer:" in text:
            return clean_answer(text.rsplit("Answer:", 1)[1])
        return clean_answer(text.strip().splitlines()[-1] if text.strip() else "")
    [(text, _)] = yield [gen(JOINT.format(pages=pages, notes=note_text, question=ctx.question), 32)]
    return clean_answer(text)


# Memory conditions: who else writes into the shared memory.
CONDITIONS = {
    "clean": {},
    "poison1": {"poison": 1},  # fabricating writer
    "poison5": {"poison": 5},  # the same claims under five Sybil identities
    "adapt1": {"poison": 1, "adaptive": True},
    "target1": {"poison": 1, "targeted": True},  # gold-aware writer pushing one wrong in-page span
    "target5": {"poison": 5, "targeted": True},
    "liar": {"liars": 1},  # the last specialist is compromised: it proposes and attests the target
}

METHODS = {
    "trustmem": (trustmem, {}),
    "tm_greedy": (trustmem, {"beam": 1}),
    "tm_max": (trustmem, {"mode": "max"}),
    "tm_verify": (trustmem, {"redemption": "verify"}),
    "tm_cited": (trustmem, {"cited_only": True}),
    "tm_writer": (trustmem, {"trust_writer": True}),
    "tm_nomix": (trustmem, {"mix": False}),
    "tm_safe": (trustmem, {"prior": True}),
    "tm_safe_greedy": (trustmem, {"prior": True, "beam": 1}),
    "tm_safe_writer": (trustmem, {"prior": True, "trust_writer": True}),
    "tm_safe_verify": (trustmem, {"prior": True, "redemption": "verify"}),
    "tm_safe_reply": (trustmem, {"prior": True, "propose": False}),
    "tm_safe_max": (trustmem, {"prior": True, "mode": "max"}),
    "tm_safe_cited": (trustmem, {"prior": True, "cited_only": True}),
    "tm_safe_nomix": (trustmem, {"prior": True, "mix": False}),
    "tm_safe_mix": (trustmem, {"prior": True, "combine": "mix"}),
    "tm_mix": (trustmem, {"combine": "mix"}),
    "tm_reply": (trustmem, {"propose": False}),
    "tmp": (trustmem_plans, {}),
    "tmc_one": (trustmem_plans, {"close": True, "n_plans": 0}),
    "tmc_vote": (trustmem_plans, {"close": True, "combine": "vote"}),
    "tmc_reply": (trustmem_plans, {"close": True, "propose": False}),
    "tmc_flat": (trustmem_plans, {"close": True, "mode": "flat"}),
    "tmc_writer": (trustmem_plans, {"close": True, "trust_writer": True}),
    "tmc_nodirect": (trustmem_plans, {"close": True, "direct": False}),
    "tmc_greedy": (trustmem_plans, {"close": True, "beam": 1}),
    "tmc": (trustmem_plans, {"close": True}),
    "tmc_safe": (trustmem_plans, {"close": True, "prior": True}),
    "tmg": (trustmem_plans, {"compose": "closed"}),
    "tmg_safe": (trustmem_plans, {"compose": "closed", "prior": True}),
    "tmp2": (trustmem_plans, {"n_plans": 2}),
    "tmp8": (trustmem_plans, {"n_plans": 8}),
    "tmp_greedy": (trustmem_plans, {"beam": 1}),
    "tmp_vote": (trustmem_plans, {"combine": "vote"}),
    "tmp_writer": (trustmem_plans, {"trust_writer": True}),
    "tmp_nodirect": (trustmem_plans, {"direct": False}),
    "tmp_safe": (trustmem_plans, {"prior": True}),
    "tmp_reply": (trustmem_plans, {"propose": False}),
    "tmp_verify": (trustmem_plans, {"redemption": "verify"}),
    "tmp_one": (trustmem_plans, {"n_plans": 0}),
    "tmp_flat": (trustmem_plans, {"mode": "flat"}),
    "tmp_cons": (trustmem_plans, {"mode": "consistent"}),
    "tmp_safe_cons": (trustmem_plans, {"mode": "consistent", "prior": True}),
    "tmp_safe_flat": (trustmem_plans, {"mode": "flat", "prior": True}),
    "plan_vote": (plan_vote, {}),
    "tmi": (trustmem_inquiry, {}),
    "tmi_greedy": (trustmem_inquiry, {"beam": 1}),
    "tmi_writer": (trustmem_inquiry, {"trust_writer": True}),
    "tmi_reply": (trustmem_inquiry, {"propose": False}),
    "tmi_verify": (trustmem_inquiry, {"redemption": "verify"}),
    "closed_book": (closed_book, {}),
    "plan_greedy": (plan_greedy, {}),
    "self_ask": (self_ask, {}),
    "ircot": (ircot, {}),
    "vote": (vote, {}),
    "blackboard": (blackboard, {}),
    "debate": (debate, {}),
    "joint": (joint, {}),
    "joint_cot": (joint, {"cot": True}),
}
