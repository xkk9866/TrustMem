"""Print where a clean TrustMem packet loses the answer span."""
from run_durable import HONEST, base_cfg, honest_peers, prepare_peers, readout
from run_trustmem import (
    admit_trustmem,
    assign_strength,
    close_contracts,
    crag_select,
    gold_recall,
    load_models,
    merge_replies,
    peer_attest,
)
from run_durable import load_slices

models = load_models("cuda:0")
_, test, _ = load_slices("hotpot", 4, 4)
cfg = base_cfg()
for ex in test[:2]:
    print("\n===", ex["id"], ex["answer"])
    peers, docs = prepare_peers(models, ex, "none", __import__("random").Random(0), 2, "random", 10, str(ex["id"]))
    for peer in peers:
        print(" peer", peer.producer, "gold", gold_recall(peer.evidence_text(), ex["answer"]), "n", len(peer.evidence))
        print("  ", peer.evidence_text()[:180].replace("\n", " | "))
    admitted, _ = admit_trustmem(models, peers, ex["question"], 10, docs, cfg, True)
    print(" admitted", len(admitted))
    for contract in admitted:
        print("  ", contract.producer, "gold", gold_recall(contract.evidence_text(), ex["answer"]), "claim", contract.claim[:40])
    replies = peer_attest(admitted, ex["question"], docs)
    print(" replies", len(replies))
    admitted = merge_replies(admitted, replies)
    assign_strength(models, admitted, ex["question"], 10, docs)
    closed, _ = close_contracts(models, admitted, ex["question"], cfg, limit=32)
    print(" closed", len(closed))
    for contract in closed:
        print("  ", contract.producer, "gold", gold_recall(contract.evidence_text(), ex["answer"]), "n", len(contract.evidence))
    pred, blob = readout(models, ex["question"], closed, 10**6)
    print(" readout", pred, "hold", gold_recall(blob, ex["answer"]))
    crag, _ = crag_select(peers, ex["question"], docs, str(ex["id"]), True)
    pred_c, blob_c = readout(models, ex["question"], crag, 10**6)
    print(" crag", pred_c, "hold", gold_recall(blob_c, ex["answer"]))
