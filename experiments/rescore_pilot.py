"""Rescore saved pilot packets under lexical-first readout."""
import pickle
from pathlib import Path

import numpy as np

from run_durable import LIVE
from run_trustmem import Contract, gold_recall, load_models, reader_from_contracts, token_f1, tokens

root = Path(__file__).resolve().parents[1]
models = load_models("cuda:1")
with (root / "results" / "packets_pilot.pkl").open("rb") as handle:
    test_packets, pool_packets, meta = pickle.load(handle)


def rank_sentences(question, memory, overlap_w):
    qvec = models.embed(question, query=True)
    q_toks = tokens(question)
    scored = []
    seen = set()
    for contract in memory:
        if not (contract.evidence or contract.claim):
            continue
        for ev in contract.evidence or [{"text": contract.claim}]:
            text = ev.get("text", "")
            if not text or text in seen:
                continue
            seen.add(text)
            overlap = (len(q_toks & tokens(text)) / len(q_toks)) if q_toks else 0.0
            vec = models.embed(text, query=False)
            cos = float(np.dot(qvec, vec))
            base = overlap_w * overlap + (1 - overlap_w) * max(cos, 0.0)
            write_vec = getattr(contract, "write_vec", None)
            if write_vec is not None:
                base *= 0.55 + 0.45 * max(float(np.dot(qvec, write_vec)), 0.0)
            scored.append((base, ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [ev for _, ev in scored[:4]]


def f1_of(question, answer, memory, overlap_w):
    sentences = rank_sentences(question, memory, overlap_w)
    packed = Contract("", sentences, "readout", "read", 0.5, 0, LIVE, "none", False)
    pred, _, _ = reader_from_contracts(models, question, [packed])
    return token_f1(pred, answer)


for overlap_w in (0.45, 0.70, 0.85):
    print("weight", overlap_w, flush=True)
    for setting in ("mixed", "unsupported", "incomplete", "conflict"):
        for method in ("crag", "crag_plus", "trustmem"):
            scores = []
            interference = pool_packets["clean"][method]
            for index, stored in enumerate(test_packets[setting][method]):
                start = index % len(interference)
                ordered = interference[start:] + interference[:start]
                mem = list(stored) + [c for packet in ordered[:100] for c in packet]
                scores.append(f1_of(meta[index]["question"], meta[index]["answer"], mem, overlap_w))
            print(f"  {setting:12s} {method:10s} {sum(scores)/len(scores):.3f}", flush=True)
