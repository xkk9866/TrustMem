import pickle
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name in ("hotpot", "2wiki"):
    with (root / "results" / f"packets_{name}.pkl").open("rb") as handle:
        _, _, meta = pickle.load(handle)
    print("\n==", name)
    for item in meta[:12]:
        print("-", item["answer"], "||", item["question"])
