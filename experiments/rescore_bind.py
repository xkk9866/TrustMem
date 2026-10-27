"""Rescore saved packets by reading only the write whose question is nearest."""
import pickle
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)

from run_durable import LIVE
from run_trustmem import Contract, gold_recall, load_models, reader_from_contracts, token_f1, tokens

root = Path("results")
models = load_models("cuda:1")


def rows_of(contracts):
    out = []
    seen = set()
    for contract in contracts:
        for ev in contract.evidence or [{"text": contract.claim or "", "title": ""}]:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


def f1_packet(question, answer, contracts):
    evs = rows_of(contracts)
    if not evs:
        return 0.0
    q = tokens(question)
    evs = sorted(evs, key=lambda ev: len(q & tokens(ev.get("text", ""))), reverse=True)
    packed = Contract("", evs[:4], "readout", "read", 0.5, 0, LIVE, "none", False)
    pred, _, _ = reader_from_contracts(models, question, [packed])
    return token_f1(pred, answer)


def nearest(question, packets):
    qvec = models.embed(question, query=True)
    best = None
    best_score = -1.0
    for packet in packets:
        if not packet:
            continue
        write_vec = getattr(packet[0], "write_vec", None)
        if write_vec is None:
            continue
        score = float(np.dot(qvec, write_vec))
        if score > best_score:
            best_score = score
            best = packet
    return best or []


def main():
    with (root / "packets_cite-hotpot.pkl").open("rb") as handle:
        test_packets, pool_packets, meta = pickle.load(handle)
    for setting in ("mixed", "unsupported", "incomplete", "conflict"):
        for method in ("marag", "crag", "crag_plus", "trustmem"):
            immediate = []
            late = []
            interference = pool_packets["clean"][method]
            for index, stored in enumerate(test_packets[setting][method]):
                question = meta[index]["question"]
                answer = meta[index]["answer"]
                immediate.append(f1_packet(question, answer, stored))
                start = index % len(interference)
                ordered = interference[start:] + interference[:start]
                chosen = nearest(question, [stored] + ordered[:1000])
                late.append(f1_packet(question, answer, chosen))
            print(
                f"{setting:12s} {method:10s} F1 {sum(immediate)/len(immediate):.3f} H1000 {sum(late)/len(late):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
