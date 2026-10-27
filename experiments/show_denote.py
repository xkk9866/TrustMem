import json

d = json.load(open("results/denote_read.json", encoding="utf-8"))
cols = ["marag", "crag", "faithful", "open", "chain", "gated"]
for name, ds in d["datasets"].items():
    print("==", name)
    for setting, block in ds["settings"].items():
        shown = " ".join(f"{col} {block[col]:.4f} ({block[col]:.2f})" for col in cols)
        print(setting, shown, block["fired"])
        for key, value in block["delta"].items():
            if key.startswith("gated-") or key.startswith("open-"):
                print(f"  {key} {value['mean']:+.4f} [{value['lo']:+.4f}, {value['hi']:+.4f}]")
