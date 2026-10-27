"""Reader F1 after dropping pages that share no proper name with a second page."""
import pickle
import re
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_durable import LIVE, bootstrap_delta
from run_trustmem import Contract, gold_recall, load_models, reader_from_contracts, token_f1, tokens

root = Path(__file__).resolve().parents[1]
models = load_models("cuda:0")
NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b|\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b")


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


def keep_pages(question, evs):
    q = tokens(question)
    by = {}
    for ev in evs:
        by.setdefault(ev.get("title") or ev["text"][:32], []).append(ev)
    name_pages = {}
    blobs = {}
    for title, group in by.items():
        blob = " ".join(f"{title} {ev.get('text', '')}" for ev in group)
        blobs[title] = blob
        found = set(NAME.findall(blob))
        found.add(title.split("(")[0].strip())
        for name in found:
            if len(name) < 4 or tokens(name) <= q:
                continue
            name_pages.setdefault(name.lower(), set()).add(title)
    cross = {name for name, pages in name_pages.items() if len(pages) >= 2}
    kept = []
    for title, group in by.items():
        head = tokens(title.split("(")[0])
        blob = blobs[title].lower()
        if (head & q) or any(name in blob for name in cross):
            kept.extend(group)
    return kept or evs


def f1(question, answer, evs):
    packed = Contract("", evs, "readout", "read", 0.5, 0, LIVE, "none", False)
    pred, _, _ = reader_from_contracts(models, question, [packed])
    return token_f1(pred, answer)


for dataset in ("hotpot", "2wiki"):
    with (root / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    print("\n==", dataset, flush=True)
    for setting in ("mixed", "unsupported", "incomplete"):
        scores = {}
        for method in ("crag_plus", "trustmem"):
            vals = []
            for index in range(len(meta)):
                question = meta[index]["question"]
                answer = meta[index]["answer"]
                vals.append(f1(question, answer, keep_pages(question, rows(packets[setting][method][index]))))
            scores[method] = vals
            print(f"  {setting:12s} {method:10s} {sum(vals)/len(vals):.3f}", flush=True)
        stat = bootstrap_delta(scores["trustmem"], scores["crag_plus"])
        print(f"  {setting:12s} delta {stat['delta']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}]", flush=True)
