import json

d = json.load(open("results/sota_read.json", encoding="utf-8"))
cols = ["marag", "crag", "faithful", "algebra", "best", "pages"]
for name, ds in d["datasets"].items():
    print("==", name)
    for setting, block in ds["settings"].items():
        shown = " ".join(f"{col} {block[col]:.4f} ({block[col]:.2f})" for col in cols)
        print(setting, shown)
        for key, value in block["delta"].items():
            if key.startswith("best-") or key == "algebra-faithful":
                print(
                    f"  {key} {value['mean']:+.4f} [{value['lo']:+.4f}, {value['hi']:+.4f}]"
                )
    horizon = ds.get("horizon")
    if horizon:
        print("later", horizon["later"])
        for setting, block in horizon["settings"].items():
            print(
                setting,
                f"stored {block['stored']:.4f} ({block['stored']:.2f})",
                f"commitment {block['commitment']:.4f} ({block['commitment']:.2f})",
                f"first {block['first']}",
                f"{block['delta']['mean']:+.4f} [{block['delta']['lo']:+.4f}, {block['delta']['hi']:+.4f}]",
            )
