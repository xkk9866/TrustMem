"""Read only sentences that share a proper name with a second page."""
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


def cross_sentences(evs, question):
    q = tokens(question)
    by_page = {}
    for ev in evs:
        by_page.setdefault(ev.get("title") or ev["text"][:40], []).append(ev)
    name_pages = {}
    for title, group in by_page.items():
        found = set()
        for ev in group:
            found.add(title)
            found.update(NAME.findall(ev.get("text") or ""))
        for name in found:
            if tokens(name) <= q or len(name) < 6:
                continue
            name_pages.setdefault(name.lower(), set()).add(title)
    cross = {name for name, pages in name_pages.items() if len(pages) >= 2}
    if not cross:
        return evs
    kept = []
    for ev in evs:
        blob = f"{ev.get('title', '')} {ev.get('text', '')}".lower()
        if any(name in blob for name in cross):
            kept.append(ev)
    return kept or evs


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    for method in ("crag_plus", "trustmem"):
        scores = []
        for index in range(len(meta)):
            question = meta[index]["question"]
            evs = cross_sentences(rows(packets[setting][method][index]), question)
            packed = Contract("", evs[:4], "readout", "read", 0.5, 0, LIVE, "none", False)
            pred, _, _ = reader_from_contracts(models, question, [packed])
            scores.append(token_f1(pred, meta[index]["answer"]))
        print(f"{setting:12s} {method:10s} {sum(scores)/len(scores):.3f}", flush=True)
