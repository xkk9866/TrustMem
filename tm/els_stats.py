"""Extra numbers for the journal version: partition seeds, question types, cost.

python els_stats.py  ->  results/els_stats.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_analysis as A  # noqa: E402

R = Path(__file__).resolve().parent / "results"
BASE = "Qwen2.5-7B-Instruct-AWQ"
SEED_METHODS = ["vote", "blackboard", "debate", "self_ask", "trustmem", "tmp"]


def main():
    rows = A.load([R / "final.jsonl", R / "seeds.jsonl", R / "ablate.jsonl", R / "quorum.jsonl", R / "liar.jsonl"])
    idx = A.index(rows)
    out = {"seeds": {}, "seed_delta": {}, "types": {}, "quorum": {}, "cost": {}}
    # partition seeds 0, 1, 2 (seed 0 = main run)
    for s in (0, 1, 2):
        for m in SEED_METHODS:
            for c in ("clean", "target1"):
                cells = [A.cell(idx, BASE, d, m, c, 2, s) for d in A.DATASETS]
                if all(cells):
                    out["seeds"][f"{m}|{c}|{s}"] = {k: float(np.mean([e[k] for e in cells])) for k in ("f1", "asr", "flip")}
        for b in ("self_ask", "debate"):
            for c in ("clean", "target1"):
                p = A.paired(idx, BASE, "tmp", b, A.DATASETS, [c], 2, s)
                if p:
                    out["seed_delta"][f"{b}|{c}|{s}"] = p
    # question types, seed 0, clean and targeted
    for m in ["vote", "blackboard", "debate", "ircot", "self_ask", "plan_greedy", "joint_cot", "tmp"]:
        for c in ("clean", "target1"):
            acc = defaultdict(list)
            for d in A.DATASETS:
                for r in idx.get((BASE, d, m, c, 2, 0), {}).values():
                    t = A.GOLD[(d, r["id"])]["type"]
                    acc[f"{d}|{t}"].append(r["f1"])
            out["types"][f"{m}|{c}"] = {k: [float(np.mean(v)), len(v)] for k, v in acc.items()}
    # quorum and liar summaries, flip included
    for k in ("2", "4r2", "6r3"):
        for m in ["vote", "blackboard", "debate", "ircot", "self_ask", "plan_greedy", "tmp", "tmp_flat", "tmp_safe", "joint_cot", "tmp_writer"]:
            for c in ("clean", "target1", "liar"):
                kk = 2 if k == "2" else k
                cells = [A.cell(idx, BASE, d, m, c, kk, 0) for d in A.DATASETS]
                if all(cells) and (k != "2" or c == "liar"):
                    out["quorum"][f"{k}|{m}|{c}"] = {x: float(np.mean([e[x] for e in cells])) for x in ("f1", "asr", "flip", "n")}
    for m in ["vote", "blackboard", "debate", "ircot", "self_ask", "plan_greedy", "plan_vote", "joint_cot", "trustmem", "tmp", "tmp_safe"]:
        cells = [A.cell(idx, BASE, d, m, "clean", 2, 0) for d in A.DATASETS]
        if all(cells):
            out["cost"][m] = {x: float(np.mean([e[x] for e in cells])) for x in ("gen", "score")}
    (R / "els_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=0)[:20000])


if __name__ == "__main__":
    main()
