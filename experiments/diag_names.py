"""Find whether the overview examples are in the stored test packets."""
import pickle
from pathlib import Path

path = Path("results/packets_2wiki.pkl")
with path.open("rb") as handle:
    packets, pool, meta = pickle.load(handle)
print("meta", len(meta), "keys", sorted(meta[0].keys()))
for item in meta:
    question = item.get("question", "")
    if "Gallivant" in question or "Summer" in question or "Deadly" in question or "Years" in question:
        print("---")
        print(question)
        print("answer:", item.get("answer"))
