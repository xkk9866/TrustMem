"""Drop a CRAG+ sentence only when it is not the question page and names no new person."""
import pickle
import re
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)

from run_durable import LIVE
from run_trustmem import Contract, gold_recall, load_models, reader_from_contracts, token_f1, tokens

root = Path("results")
with (root / "packets_cite120.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
models = load_models("cuda:1")
NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


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


def protected(ev, question):
    q = tokens(question)
    head = tokens((ev.get("title") or "").split("(")[0])
    if head & q:
        return True
    for name in NAME.findall(ev.get("text") or ""):
        if tokens(name) - q:
            return True
    return False


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    scores = []
    for index in range(len(meta)):
        question = meta[index]["question"]
        base = rows(packets[setting]["crag_plus"][index])
        seen = {ev["text"] for ev in base}
        extra = [ev for ev in rows(packets[setting]["trustmem"][index]) if ev["text"] not in seen]
        kept = [ev for ev in base if protected(ev, question)]
        evidence = (kept or base) + extra
        packed = Contract("", evidence[:6], "readout", "read", 0.5, 0, LIVE, "none", False)
        pred, _, _ = reader_from_contracts(models, question, [packed])
        scores.append(token_f1(pred, meta[index]["answer"]))
    print(f"{setting:12s} {sum(scores)/len(scores):.3f}", flush=True)
