"""See how producers and pages are stored in the paper packets."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_hotpot.pkl").open("rb") as handle:
    packets, pool, meta = pickle.load(handle)

stored = packets["mixed"]["trustmem"][0]
print("question", meta[0]["question"])
print("n contracts", len(stored))
for contract in stored:
    print(
        " producer",
        contract.producer,
        "n_ev",
        len(contract.evidence or []),
        "fault",
        contract.fault,
        "claim",
        (contract.claim or "")[:40],
    )
    for ev in (contract.evidence or [])[:3]:
        print("   ", ev.get("title", "")[:40], "|", (ev.get("text") or "")[:80])

print("--- crag_plus")
stored = packets["mixed"]["crag_plus"][0]
print("n contracts", len(stored))
for contract in stored:
    print(" producer", contract.producer, "n_ev", len(contract.evidence or []))
    for ev in (contract.evidence or [])[:6]:
        print("   ", ev.get("title", "")[:40], "|", (ev.get("text") or "")[:80])
