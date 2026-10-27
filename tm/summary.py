"""Quick F1 / ASR grid: python summary.py results/pilot7.jsonl [more.jsonl ...]"""
import collections
import json
import sys

rows = {}
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows[r["key"]] = r
g = collections.defaultdict(list)
for r in rows.values():
    g[(r["dataset"], r["cond"], r["method"], r["k"], r["seed"])].append(r)
conds = ["clean", "poison1", "poison5", "adapt1", "target1", "target5", "liar"]
for d in sorted({k[0] for k in g}):
    for kk, s in sorted({(k[3], k[4]) for k in g if k[0] == d}):
        print(f"== {d} k{kk} s{s}")
        ms = sorted({k[2] for k in g if k[0] == d and k[3] == kk and k[4] == s})
        for m in ms:
            line = f"{m:16s}"
            for c in conds:
                L = g.get((d, c, m, kk, s), [])
                if L:
                    f = sum(r["f1"] for r in L) / len(L)
                    a = sum(1 for r in L if r["adv"] >= 0.5 and r["f1"] < 0.5) / len(L)
                    line += f" {c[:6]} {f:.3f}({100 * a:4.1f}) n{len(L):<4d}"
            gen = sum(r["n_gen"] for r in g[(d, "clean", m, kk, s)]) / max(1, len(g[(d, "clean", m, kk, s)]))
            print(line, f"gen {gen:.1f}")
