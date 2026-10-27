"""Reader F1 if TrustMem adds the hop sentence and never drops a CRAG+ sentence."""
import json
import pickle
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)

from run_durable import LIVE
from run_trustmem import Contract, load_models, reader_from_contracts, token_f1

root = Path("results")
with (root / "packets_cite60b.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
models = load_models("cuda:1")


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    scores = []
    for index in range(len(meta)):
        base = rows(packets[setting]["crag_plus"][index])
        seen = {ev["text"] for ev in base}
        extra = [ev for ev in rows(packets[setting]["trustmem"][index]) if ev["text"] not in seen]
        packed = Contract("", base + extra[:2], "readout", "read", 0.5, 0, LIVE, "none", False)
        pred, _, _ = reader_from_contracts(models, meta[index]["question"], [packed])
        scores.append(token_f1(pred, meta[index]["answer"]))
    print(f"{setting:12s} union={sum(scores)/len(scores):.3f}", flush=True)
