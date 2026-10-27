"""What does Contract.owned hold?"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_hotpot.pkl").open("rb") as handle:
    packets, _, _ = pickle.load(handle)
contract = packets["unsupported"]["trustmem"][0][0]
print("owned type", type(contract.owned), "len", len(contract.owned or ()))
item = (contract.owned or [None])[0]
print("item type", type(item))
if isinstance(item, dict):
    print(item.keys())
    print("title", item.get("title"))
    print("text", (item.get("text") or item.get("sentences") or "")[:200] if isinstance(item.get("text") or "", str) else type(item.get("text")))
elif isinstance(item, (list, tuple)):
    print("tuple len", len(item), "first", str(item[0])[:80] if item else None)
else:
    print(str(item)[:300])
print("ev keys", contract.evidence[0].keys())
print("fault", contract.fault, "claim", contract.claim[:80])
