import json

d = json.load(open(r"results/scores_hotpot.json", encoding="utf-8"))
for key, row in d.items():
    hold = row.get("hold", {})
    extra = ""
    if "miss_n" in row:
        extra = f" miss {row['miss_n']} {row['miss_f1']:.3f}"
    print(
        f"{key:28s} f1 {row['f1']:.3f} "
        f"hold {hold.get('4', 0):.3f}/{hold.get('100', 0):.3f}/{hold.get('1000', 0):.3f}{extra}"
    )
