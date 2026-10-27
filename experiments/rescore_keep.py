"""Keep question-titled pages and sentences that introduce a new proper name. Add hop sentences."""
import pickle
import re
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)

from run_durable import LIVE
from run_trustmem import Contract, load_models, reader_from_contracts, token_f1, tokens

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


def keep_base(ev, question):
    title = ev.get("title") or ""
    head = tokens(title.split("(")[0])
    q = tokens(question)
    if head & q:
        return True
    names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", ev.get("text") or "")
    return any(tokens(name) - q for name in names)


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    scores = []
    changed = 0
    for index in range(len(meta)):
        question = meta[index]["question"]
        base = rows(packets[setting]["crag_plus"][index])
        seen = {ev["text"] for ev in base}
        extra = [ev for ev in rows(packets[setting]["trustmem"][index]) if ev["text"] not in seen][:2]
        kept = [ev for ev in base if keep_base(ev, question)]
        evidence = (kept or base) + extra
        if [ev["text"] for ev in evidence] != [ev["text"] for ev in base]:
            changed += 1
        packed = Contract("", evidence[:6], "readout", "read", 0.5, 0, LIVE, "none", False)
        pred, _, _ = reader_from_contracts(models, question, [packed])
        scores.append(token_f1(pred, meta[index]["answer"]))
    print(f"{setting:12s} keep={sum(scores)/len(scores):.3f} changed={changed}", flush=True)
