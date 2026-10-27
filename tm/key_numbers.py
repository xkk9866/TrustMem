"""Numbers quoted in the text of the paper.  python key_numbers.py results/*.jsonl"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_analysis as A  # noqa: E402

B = "Qwen2.5-7B-Instruct-AWQ"
idx = A.index(A.load(sys.argv[1:]))
out = {}


def P(a, b, ds, cs, key="f1", k=2):
    p = A.paired(idx, B, a, b, ds, cs, k=k, key=key)
    return None if p is None else [round(x, 3) for x in p[:3]] + [p[3]]


for m in ["tmp", "trustmem", "tmp_writer", "self_ask", "debate", "plan_vote", "plan_greedy", "joint_cot", "ircot", "blackboard", "vote", "tmp_safe"]:
    es = [A.cell(idx, B, d, m, "clean") for d in A.DATASETS]
    if all(es):
        out[f"cost|{m}"] = [round(np.mean([e["gen"] for e in es]), 1), round(np.mean([e["score"] for e in es]), 1)]
for c in A.CONDS + ["attacked"]:
    cs = A.ATTACKS if c == "attacked" else [c]
    out[f"vs_jointcot|{c}"] = P("tmp", "joint_cot", A.DATASETS, cs)
out["vs_jointcot|all"] = P("tmp", "joint_cot", A.DATASETS, A.CONDS)
for d in A.DATASETS:
    out[f"vs_single|{d}"] = P("tmp", "trustmem", [d], A.CONDS)
    out[f"vs_single_clean|{d}"] = P("tmp", "trustmem", [d], ["clean"])
    for m in ["debate", "self_ask", "blackboard"]:
        out[f"vs_{m}_clean|{d}"] = P("tmp", m, [d], ["clean"])
out["vs_best_baseline_clean"] = {m: P("tmp", m, A.DATASETS, ["clean"]) for m in ["self_ask", "debate", "plan_greedy", "tmp_writer"]}
out["vs_safe"] = P("tmp", "tmp_safe", A.DATASETS, ["clean"])
for m in ["tmp_vote", "tmp_one", "tmp_reply", "tmp_greedy", "tmp_nodirect", "tmp_verify", "tmp2", "tmp8", "tmp_writer"]:
    out[f"abl|{m}"] = P("tmp", m, A.DATASETS, ["clean"])
    out[f"abl_t|{m}"] = P("tmp", m, A.DATASETS, ["target1"])
# attack drop: attacked minus clean F1, per method
for m in ["tmp", "joint_cot", "self_ask", "debate", "blackboard", "vote", "ircot", "plan_greedy", "tmp_writer"]:
    drops = []
    for d in A.DATASETS:
        c = A.cell(idx, B, d, m, "clean")
        for a in A.ATTACKS:
            e = A.cell(idx, B, d, m, a)
            if c and e:
                drops.append(e["f1"] - c["f1"])
    if drops:
        out[f"drop|{m}"] = round(float(np.mean(drops)), 3)
out["separation"] = A.claim_separation(idx, B)
out["invariance"] = A.invariance(idx, B, ["tmp", "trustmem", "tmp_writer", "self_ask", "debate", "blackboard", "vote", "ircot", "joint_cot", "plan_greedy"])
out["attack_wins"] = {m: A.where_attacks_win(idx, B, m) for m in ("tmp", "tmp_safe")}
out["plans"] = A.plan_stats(idx, B)
# oracle over plan answers
orc = {}
for d in A.DATASETS:
    rs = list(idx.get((B, d, "tmp", "clean", 2, 0), {}).values())
    pv = list(idx.get((B, d, "plan_vote", "clean", 2, 0), {}).values())
    if rs:
        g = lambda r: A.GOLD[(d, r["id"])]["golds"]
        orc[d] = {"tmp_oracle": round(np.mean([max(max(A.f1(p["top"], x) for x in g(r)) for p in r["log"]["plans"]) for r in rs]), 3),
                  "tmp": round(np.mean([r["f1"] for r in rs]), 3)}
        if pv:
            orc[d]["planvote_oracle"] = round(np.mean([max(max(A.f1(a or "", x) for x in g(r)) for a in r["log"]["answers"]) for r in pv]), 3)
out["oracle"] = orc
print(json.dumps(out, indent=1))
Path("results/key_numbers.json").write_text(json.dumps(out, indent=1))
