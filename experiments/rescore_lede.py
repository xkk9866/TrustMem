"""Add the identity sentence of an opened page to the corrective read.

An identity sentence names its page and gives the definitional fact (title lead,
or a parenthetical life span). Comparison questions are also read from identity
sentences alone. The corrective sentences stay for every other question.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_trustmem import Contract, load_models, normalize_answer, reader_from_contracts, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")
YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
CUES = (
    "died",
    "born",
    "more recent",
    "earlier",
    "later",
    "director",
    "composer",
    "grandfather",
    "grandmother",
    "where was",
    "which country",
    "paternal",
    "maternal",
    "which film",
    "who died",
    "who was born",
)


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


def rank_top(question, evs, k=4):
    if not evs:
        return []
    q_toks = tokens(question)
    qvec = MODELS.embed(question, query=True)
    scored = []
    for ev in evs:
        text = ev.get("text") or ""
        overlap = (len(q_toks & tokens(text)) / len(q_toks)) if q_toks else 0.0
        cos = float(np.dot(qvec, MODELS.embed(text, query=False)))
        scored.append((0.45 * overlap + 0.55 * max(cos, 0.0), ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [ev for _, ev in scored[:k]]


def answer(question, evs):
    if not evs:
        return ""
    packed = Contract("", evs, "readout", "read", 0.5, 0, 10**9, "none", False)
    pred, _, _ = reader_from_contracts(MODELS, question, [packed])
    return pred


def is_identity(ev):
    title = (ev.get("title") or "").split("(")[0].strip()
    text = ev.get("text") or ""
    if title and text.lower().startswith(title.lower()[: max(8, min(len(title), 24))].strip()):
        return True
    if "(" in text[:180] and YEAR.search(text[:180]):
        return True
    return False


def identity_sentences(question, mem, plus):
    have = {ev.get("text") for ev in plus}
    q = question.lower()
    picked = []
    for ev in mem:
        if ev.get("text") in have or not is_identity(ev):
            continue
        title = (ev.get("title") or "").split("(")[0].strip().lower()
        named = len(title) > 3 and title in q
        if named:
            picked.append(ev)
    return picked[:2]


def carries(text, gold):
    needle = normalize_answer(gold)
    return bool(needle) and needle not in {"yes", "no"} and needle in normalize_answer(text)


def route_context(question, mem, plus4):
    q = " " + question.lower() + " "
    if any(cue in q for cue in CUES):
        ident = [ev for ev in mem if is_identity(ev)]
        # Prefer identity sentences whose page is named by the question.
        named = []
        other = []
        for ev in ident:
            title = (ev.get("title") or "").split("(")[0].strip().lower()
            if len(title) > 3 and title in question.lower():
                named.append(ev)
            else:
                other.append(ev)
        chosen = (named or other)[:4]
        if chosen:
            return chosen
    extra = identity_sentences(question, mem, plus4)
    return plus4 + extra


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "conflict", "clean"):
            scores = {name: [] for name in ("crag_plus", "trustmem", "graft", "route", "oracle")}
            routed = 0
            for item, index in zip(meta, range(len(meta))):
                question = item["question"]
                gold = item["answer"]
                plus = rank_top(question, rows(packets[setting]["crag_plus"][index]))
                mem = rows(packets[setting]["trustmem"][index])
                mem4 = rank_top(question, mem)
                grafted = plus + identity_sentences(question, mem, plus)
                ctx = route_context(question, mem, plus)
                q = " " + question.lower() + " "
                if any(cue in q for cue in CUES):
                    routed += 1
                scores["crag_plus"].append(token_f1(answer(question, plus), gold))
                scores["trustmem"].append(token_f1(answer(question, mem4), gold))
                scores["graft"].append(token_f1(answer(question, grafted), gold))
                scores["route"].append(token_f1(answer(question, ctx), gold))
                oracle = [ev for ev in mem if carries(ev.get("text") or "", gold)][:3]
                scores["oracle"].append(token_f1(answer(question, oracle or plus), gold))
            print(
                f"  {setting:12s} plus {np.mean(scores['crag_plus']):.3f}"
                f"  mem {np.mean(scores['trustmem']):.3f}"
                f"  graft {np.mean(scores['graft']):.3f}"
                f"  route {np.mean(scores['route']):.3f}"
                f"  oracle {np.mean(scores['oracle']):.3f}"
                f"  cues {routed}/{len(meta)}",
                flush=True,
            )
            for name in ("graft", "route"):
                for base in ("crag_plus", "trustmem"):
                    delta, lo, hi = bootstrap(scores[name], scores[base])
                    print(f"    {name}-{base}: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
