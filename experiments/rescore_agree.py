"""Read only sentences that share a name with a second page. Old session-log packets."""
import pickle
import re
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_durable import LIVE
from run_trustmem import Contract, load_models, reader_from_contracts, token_f1, tokens

root = Path(__file__).resolve().parents[1]
models = load_models("cuda:0")
NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b|\b(?:1|2)\d{3}\b")


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


def agreed(evs, question):
    q = tokens(question)
    by_title = {}
    for ev in evs:
        by_title.setdefault(ev.get("title") or ev["text"][:32], []).append(ev)
    name_pages = {}
    for title, group in by_title.items():
        found = {title.split("(")[0].strip()}
        for ev in group:
            found.update(NAME.findall(ev.get("text") or ""))
        for name in found:
            name = name.strip()
            if len(name) < 4 or tokens(name) <= q:
                continue
            name_pages.setdefault(name.lower(), set()).add(title)
    cross = {name for name, pages in name_pages.items() if len(pages) >= 2}
    if not cross:
        return evs, False
    kept = []
    for ev in evs:
        blob = f"{ev.get('title', '')} {ev.get('text', '')}".lower()
        if any(name in blob for name in cross):
            kept.append(ev)
    return (kept or evs), bool(kept)


def f1(question, answer, evs):
    packed = Contract("", evs[:6], "readout", "read", 0.5, 0, LIVE, "none", False)
    pred, _, _ = reader_from_contracts(models, question, [packed])
    return token_f1(pred, answer)


for dataset in ("hotpot", "2wiki"):
    with (root / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    print("\n==", dataset, flush=True)
    for setting in ("mixed", "unsupported", "incomplete", "conflict"):
        if setting not in packets:
            continue
        for method in ("crag_plus", "trustmem"):
            joint, agree, changed = [], [], 0
            for index, stored in enumerate(packets[setting][method]):
                question = meta[index]["question"]
                answer = meta[index]["answer"]
                evs = rows(stored)
                joint.append(f1(question, answer, evs))
                kept, did = agreed(evs, question)
                if did and [e["text"] for e in kept] != [e["text"] for e in evs]:
                    changed += 1
                agree.append(f1(question, answer, kept))
            n = len(joint)
            print(
                f"  {setting:12s} {method:10s} joint={sum(joint)/n:.3f} agree={sum(agree)/n:.3f} changed={changed}",
                flush=True,
            )
