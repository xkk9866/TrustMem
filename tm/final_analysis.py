"""All numbers in the paper, from the raw result logs.

python final_analysis.py results/final.jsonl results/ablate.jsonl results/seeds.jsonl results/scale.jsonl ...
Writes results/report.json and prints a readable summary.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import f1, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATASETS = ["hotpot", "2wiki", "musique"]
ATTACKS = ["poison1", "poison5", "target1"]
CONDS = ["clean"] + ATTACKS
MULTI = ["vote", "blackboard", "debate", "ircot", "self_ask", "plan_greedy", "plan_vote", "closed_book"]
FOCUS = "tmp"


def gold_table() -> dict:
    gold = {}
    for d in DATASETS + ["case"]:
        path = ROOT / "data_new" / f"{d}.jsonl"
        for line in open(path, encoding="utf-8"):
            it = json.loads(line)
            gold[(d, it["id"])] = {"golds": [it["answer"], *it.get("aliases", [])], "inter": it.get("inter", []), "type": it["type"]}
    return gold


GOLD = gold_table()


def wrong_value(row: dict, v: str) -> bool:
    golds = GOLD[(row["dataset"], row["id"])]["golds"]
    return bool(normalize(v)) and max(f1(v, g) for g in golds) < 0.5


def attacked(row: dict) -> float:
    """The answer was taken from an untrusted writer: it matches a planted value that is itself wrong."""
    if row["cond"] == "clean" or row["f1"] >= 0.5:
        return 0.0
    vals = [v for v in row["log"].get("poison_values", []) if wrong_value(row, v)]
    return float(any(f1(row["pred"], v) >= 0.5 for v in vals))


def load(paths) -> list[dict]:
    rows = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            r["asr"] = attacked(r)
            if r.get("mirror", 1) > 1:
                r["k"] = f"{r['k']}r{r['mirror']}"
            rows[r["key"]] = r
    return list(rows.values())


def index(rows):
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["model"], r["dataset"], r["method"], r["cond"], r["k"], r["seed"])][r["id"]] = r
    return idx


def boot(d: np.ndarray, n: int = 2000, seed: int = 3152):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def cell(idx, model, d, m, c, k=2, s=0):
    by = idx.get((model, d, m, c, k, s), {})
    if not by:
        return None
    v = list(by.values())
    clean = idx.get((model, d, m, "clean", k, s), {})
    good = [r for r in v if r["id"] in clean and clean[r["id"]]["f1"] >= 0.5]
    flip = float(np.mean([r["asr"] for r in good])) if good and c != "clean" else 0.0
    return {"n": len(v), "flip": flip, "f1": float(np.mean([r["f1"] for r in v])), "em": float(np.mean([r["em"] for r in v])),
            "asr": float(np.mean([r["asr"] for r in v])), "gen": float(np.mean([r["n_gen"] for r in v])),
            "score": float(np.mean([r["n_score"] for r in v]))}


def paired(idx, model, a, b, ds, cs, k=2, s=0, key="f1"):
    da = []
    for d in ds:
        for c in cs:
            A, B = idx.get((model, d, a, c, k, s), {}), idx.get((model, d, b, c, k, s), {})
            ids = sorted(set(A) & set(B))
            da += [A[i][key] - B[i][key] for i in ids]
    if not da:
        return None
    return boot(np.array(da)) + (len(da),)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    r1 = ranks[: len(pos)].sum()
    return float((r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def claim_separation(idx, model, method=FOCUS):
    """Can a signal tell planted (wrong) values from honest correct values?  Positives: honest
    claims whose value is a gold intermediate or final answer.  Negatives: planted wrong values."""
    out = {}
    for c in ATTACKS:
        pos, neg = defaultdict(list), defaultdict(list)
        for d in DATASETS:
            for r in idx.get((model, d, method, c, 2, 0), {}).values():
                g = GOLD[(d, r["id"])]
                truth = g["golds"] + g["inter"]
                for cl in r["log"].get("claims", []):
                    good = cl["honest"] and not cl["poison"] and max(f1(cl["v"], t) for t in truth) >= 0.5
                    bad = cl["poison"] and not cl["honest"] and wrong_value(r, cl["v"]) and max(f1(cl["v"], t) for t in truth) < 0.5
                    if not (good or bad):
                        continue
                    for sig in ("support", "trust", "conf", "n_writers"):
                        (pos if good else neg)[sig].append(float(cl[sig]))
        out[c] = {sig: auroc(pos[sig], neg[sig]) for sig in ("support", "trust", "conf", "n_writers")}
        out[c]["n_pos"], out[c]["n_neg"] = len(pos["support"]), len(neg["support"])
        out[c]["planted_trust_mean"] = float(np.mean(neg["trust"])) if neg["trust"] else float("nan")
        out[c]["honest_trust_mean"] = float(np.mean(pos["trust"])) if pos["trust"] else float("nan")
    return out


def invariance(idx, model, methods):
    """Share of questions whose answer is identical under one untrusted writer and five copies of it."""
    res = {}
    for m in methods:
        same = tot = 0
        for d in DATASETS:
            A, B = idx.get((model, d, m, "poison1", 2, 0), {}), idx.get((model, d, m, "poison5", 2, 0), {})
            for i in set(A) & set(B):
                tot += 1
                same += normalize(A[i]["pred"]) == normalize(B[i]["pred"])
        if tot:
            res[m] = same / tot
    return res


def where_attacks_win(idx, model, method=FOCUS, cond="target1"):
    """Attack successes against the unresolved mass: Proposition 2 says they live where the memory holds no supported value."""
    win, lose = [], []
    for d in DATASETS:
        for r in idx.get((model, d, method, cond, 2, 0), {}).values():
            u = r["log"].get("unresolved", 0.0)
            tot = u + sum(r["log"].get("belief", {}).values())
            share = u / tot if tot > 0 else 1.0
            (win if r["asr"] else lose).append(share)
    if not win:
        return {"n_win": 0}
    return {"n_win": len(win), "n": len(win) + len(lose), "win_unresolved_median": float(np.median(win)),
            "win_frac_unresolved_gt_half": float(np.mean([w > 0.5 for w in win])),
            "rest_frac_unresolved_gt_half": float(np.mean([w > 0.5 for w in lose]))}


def plan_stats(idx, model, method=FOCUS):
    """How often the evidence moves belief off the composer's greedy plan."""
    out = {}
    for d in DATASETS:
        moved = tot = 0
        nplans = []
        for r in idx.get((model, d, method, "clean", 2, 0), {}).values():
            plans = r["log"].get("plans", [])
            if not plans:
                continue
            tot += 1
            nplans.append(len(plans))
            first = plans[0]["top"]
            moved += normalize(first) != normalize(r["pred"])
        if tot:
            out[d] = {"distinct_plans": float(np.mean(nplans)), "answer_differs_from_greedy_plan": moved / tot}
    return out


def main(paths):
    rows = load(paths)
    idx = index(rows)
    models = sorted({k[0] for k in idx})
    report = {"cells": {}, "paired": {}, "models": models}
    for key in idx:
        model, d, m, c, k, s = key
        report["cells"]["|".join(map(str, key))] = cell(idx, *key)
    base = "Qwen2.5-7B-Instruct-AWQ"
    methods = sorted({k[2] for k in idx if k[0] == base and k[4] == 2 and k[5] == 0})
    for m in methods:
        if m == FOCUS:
            continue
        for c in CONDS:
            report["paired"][f"{m}|{c}"] = paired(idx, base, FOCUS, m, DATASETS, [c])
        report["paired"][f"{m}|all"] = paired(idx, base, FOCUS, m, DATASETS, CONDS)
        report["paired"][f"{m}|attacked"] = paired(idx, base, FOCUS, m, DATASETS, ATTACKS)
        report["paired"][f"{m}|asr_attacked"] = paired(idx, base, FOCUS, m, DATASETS, ATTACKS, key="asr")
    report["separation"] = claim_separation(idx, base)
    report["invariance"] = invariance(idx, base, methods)
    report["attack_wins"] = {m: where_attacks_win(idx, base, m) for m in (FOCUS, "trustmem", "tmp_safe")}
    report["plans"] = plan_stats(idx, base)
    # seeds and team size
    seeds = defaultdict(dict)
    for (model, d, m, c, k, s), by in idx.items():
        if model == base and k == 2:
            seeds[(d, m, c)][s] = float(np.mean([r["f1"] for r in by.values()]))
    report["seeds"] = {"|".join(k): v for k, v in seeds.items() if len(v) > 1}
    for other in models:
        if other != base:
            report.setdefault("backbone", {})[other] = {
                f"{d}|{m}|{c}": cell(idx, other, d, m, c) for d in DATASETS for m in methods + [FOCUS] for c in CONDS
                if cell(idx, other, d, m, c)}
    out = ROOT / "results" / "report.json"
    out.write_text(json.dumps(report, indent=1))
    # readable
    for d in DATASETS:
        print(f"\n== {d}")
        for m in [FOCUS] + [x for x in methods if x != FOCUS]:
            line = f"{m:14s}"
            for c in CONDS:
                e = cell(idx, base, d, m, c)
                line += f"  {c[:7]:7s} {e['f1']:.3f}/{100 * e['asr']:4.1f} n{e['n']:<4d}" if e else f"  {c[:7]:7s}     -     "
            print(line)
    print("\nparied (all conds):")
    for m in methods:
        p = report["paired"].get(f"{m}|all")
        if p:
            print(f"  {FOCUS} - {m:14s} {p[0]:+.3f} [{p[1]:+.3f},{p[2]:+.3f}] n{p[3]}")
    print("\nseparation", json.dumps(report["separation"], indent=0))
    print("invariance", report["invariance"])
    print("attack wins", report["attack_wins"])
    print("plans", report["plans"])
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
