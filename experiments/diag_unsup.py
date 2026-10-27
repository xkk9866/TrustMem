"""Why unsupported TrustMem still trails CRAG after faithful provenance."""
import random

from run_durable import base_cfg, prepare_peers
from run_durable import admit_method
from run_trustmem import clone_contract, gold_recall, load_models
from run_durable import load_slices

models = load_models("cuda:1")
_, test, _ = load_slices("hotpot", 8, 4)
cfg = base_cfg()
for index, ex in enumerate(test):
    rng = random.Random(f"0:{ex['id']}:unsupported")
    peers, docs = prepare_peers(models, ex, "unsupported", rng, 2, "random", 10 + index, str(ex["id"]))
    print("\n===", ex["answer"])
    for peer in peers:
        print(" peer", peer.producer, peer.fault, "gold", round(gold_recall(peer.evidence_text(), ex["answer"]), 2), peer.claim[:60])
    for method in ("crag", "trustmem"):
        stored = admit_method(method, models, [clone_contract(c) for c in peers], ex["question"], 10 + index, docs, cfg, str(ex["id"]))
        blob = " ".join(c.evidence_text() for c in stored)
        print(" ", method, "n", sum(len(c.evidence) for c in stored), "gold", round(gold_recall(blob, ex["answer"]), 2))
        print("   ", blob[:220].replace("\n", " | "))
