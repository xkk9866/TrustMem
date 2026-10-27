import json
from pathlib import Path

d = json.loads(Path("results/openrag_results.json").read_text(encoding="utf-8"))
for ds in ("hotpot", "2wiki"):
    print("==", ds)
    block = d[ds]
    print("keys", list(block))
    for name, row in block["methods"].items():
        print(f"  {name:12s} f1={row['f1']:.3f} support={row.get('support', 0):.3f}")
    paired = block.get("paired") or block.get("bootstrap") or {}
    for key, val in paired.items():
        if isinstance(val, dict) and "diff" in val:
            print(f"  {key}: {val['diff']:+.3f} [{val['lo']:+.3f}, {val['hi']:+.3f}]")
