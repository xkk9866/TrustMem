import json
from pathlib import Path

p = json.loads(Path("results/attest_results.json").read_text(encoding="utf-8"))


def rnd(x):
    return f"{x:.2f}"


mx = 0.0
for block in p["datasets"].values():
    for methods in block["mean"].values():
        for stats in methods.values():
            for key in ("f1_std", "probe_f1_std", "hold_std"):
                mx = max(mx, stats[key])
print("max_std", f"{mx:.4f}")

for name in ("hotpot", "2wiki"):
    print("DATASET", name)
    for setting, methods in p["datasets"][name]["mean"].items():
        print(" ", setting)
        for method, stats in methods.items():
            print(
                f"   {method:10s} {rnd(stats['f1'])} {rnd(stats['probe_f1'])} {rnd(stats['hold'])}"
                f"  std {stats['f1_std']:.3f} {stats['probe_f1_std']:.3f} {stats['hold_std']:.3f}"
            )

print("ABLATION")
for name, settings in p["ablation"].items():
    for setting, stats in settings.items():
        print(name, setting, rnd(stats["f1"]), rnd(stats["probe_f1"]), rnd(stats["hold"]))

print("SWEEP")
for row in p["sweep"]:
    bits = []
    for method, stats in row["mixed"].items():
        bits.append(f"{method} {rnd(stats['probe_f1'])}/{rnd(stats['hold'])}")
    print(row["cap"], row["delay"], " | ".join(bits))

trace = json.loads(Path("results/attest_trace.json").read_text(encoding="utf-8"))
print("TRACE", len(trace))
best = sorted(trace, key=lambda row: (row["f1"], row["hold"]), reverse=True)[:6]
for row in best:
    print(
        round(row["f1"], 2),
        round(row["hold"], 2),
        row["fault"],
        row["answer"],
        "|",
        row["pred"][:70],
        "|",
        row["id"],
    )
    print("   Q", row["dataset_question"][:140])
    print("   planted", row["injected_claim"][:80])
