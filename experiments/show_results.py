import json
from pathlib import Path

data = json.loads(Path("results/trustmem_results.json").read_text(encoding="utf-8"))
for dataset, block in data.get("datasets", {}).items():
    print("DATASET", dataset)
    mean = block["mean"]
    for setting, methods in mean.items():
        print("==", setting)
        for method, stats in methods.items():
            print(
                f"{method:10} F1 {stats['f1']:.3f}+/-{stats['f1_std']:.3f} "
                f"EM {stats['em']:.3f}+/-{stats['em_std']:.3f} "
                f"pF1 {stats['probe_f1']:.3f} prop {stats['prop']:.3f} "
                f"adm {stats['admit']:.3f} ece {stats['ece']:.3f} "
                f"carry {stats['carry']:.3f} chk {stats['checks']:.2f} tok {stats['used_tokens']:.1f}"
            )
if "ablation" in data:
    print("ABLATION")
    for name, stats in data["ablation"].items():
        print(
            f"{name:12} F1 {stats['f1']:.3f}+/-{stats['f1_std']:.3f} "
            f"pF1 {stats['probe_f1']:.3f} prop {stats['prop']:.3f} adm {stats['admit']:.3f}"
        )
