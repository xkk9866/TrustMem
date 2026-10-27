"""How often a cited page names another owned page."""
import random

import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)

from run_durable import admit_method, base_cfg, load_slices, prepare_peers
from run_trustmem import clone_contract, gold_recall, load_models

models = load_models("cuda:1")
_, test, _ = load_slices("hotpot", 12, 4)
cfg = base_cfg()
hits = 0
for index, ex in enumerate(test):
    rng = random.Random(f"0:{ex['id']}:unsupported")
    peers, docs = prepare_peers(models, ex, "unsupported", rng, 2, "random", 10 + index, str(ex["id"]))
    stored = admit_method(
        "trustmem", models, [clone_contract(c) for c in peers], ex["question"], 10 + index, docs, cfg, str(ex["id"])
    )
    blob = " ".join(c.evidence_text() for c in stored)
    titles = []
    for contract in stored:
        for ev in contract.evidence:
            if ev.get("title") and ev["title"] not in titles:
                titles.append(ev["title"])
    mark = "GOLD" if gold_recall(blob, ex["answer"]) > 0.5 else "miss"
    print(f"{mark:4s} n={sum(len(c.evidence) for c in stored):2d} titles={titles[:6]}")
    if gold_recall(blob, ex["answer"]) > 0.5:
        hits += 1
print("gold packets", hits, "of", len(test))
