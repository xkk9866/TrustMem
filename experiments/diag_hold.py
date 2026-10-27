"""Check whether a correct Qwen answer was present in the specialist's pages."""

import random

from run_durable import load_slices, partition
from run_trustmem import normalize_answer

_, test, _ = load_slices("hotpot", 3, 1)
for ex in test:
    rng = random.Random(f"{ex['id']}:2:random")
    groups = partition(ex["docs"], 2, "random", rng)
    needle = normalize_answer(ex["answer"])
    print("\nQ:", ex["question"][:80])
    print("A:", ex["answer"])
    for index, docs in enumerate(groups):
        blob = normalize_answer(" ".join(" ".join(doc["sentences"]) for doc in docs))
        titles = [doc["title"] for doc in docs]
        print(f"  agent_{index} has_answer={needle in blob} titles={titles[:6]}")
