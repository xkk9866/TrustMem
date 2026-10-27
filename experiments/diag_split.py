"""Split the 60-question pilot by whether citation prune kept a short packet."""
import json
import pickle
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_cite60.pkl").open("rb") as handle:
    packets, _, _ = pickle.load(handle)
scores = json.loads((root / "results" / "scores_cite60.json").read_text(encoding="utf-8"))


def n_sent(stored):
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            if ev.get("text"):
                seen.add(ev["text"])
    return len(seen)


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    short = []
    long = []
    for index, stored in enumerate(packets[setting]["trustmem"]):
        row = {
            "tm": scores[f"{setting}:trustmem"]["f1s"][index],
            "plus": scores[f"{setting}:crag_plus"]["f1s"][index],
            "crag": scores[f"{setting}:crag"]["f1s"][index],
        }
        (short if n_sent(stored) <= 4 else long).append(row)
    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")
    print(
        f"{setting:12s} short n={len(short):2d} tm={mean(short,'tm'):.3f} plus={mean(short,'plus'):.3f} crag={mean(short,'crag'):.3f}"
        f" | long n={len(long):2d} tm={mean(long,'tm'):.3f} plus={mean(long,'plus'):.3f} crag={mean(long,'crag'):.3f}"
    )
