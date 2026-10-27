import json
from pathlib import Path

data = json.loads(Path("results/closure_results.json").read_text(encoding="utf-8"))
for name, block in data["datasets"].items():
    print("==", name)
    for setting, methods in block["mean"].items():
        print(setting)
        for method, stats in methods.items():
            print(
                f"  {method:10} F1 {stats['f1']:.3f}+/-{stats['f1_std']:.3f} "
                f"p {stats['probe_f1']:.3f}+/-{stats['probe_f1_std']:.3f} "
                f"H {stats['hold']:.3f}+/-{stats['hold_std']:.3f}"
            )
print("ABLATION")
for name, stats in data["ablation"].items():
    print(
        f"  {name:16} F1 {stats['f1']:.3f}+/-{stats['f1_std']:.3f} "
        f"p {stats['probe_f1']:.3f}+/-{stats['probe_f1_std']:.3f} "
        f"H {stats['hold']:.3f}+/-{stats['hold_std']:.3f}"
    )
