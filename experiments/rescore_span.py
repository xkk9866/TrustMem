"""Joint-read the sentences that state a span repeated on another page."""
import pickle
import re
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_durable import LIVE
from run_trustmem import Contract, load_models, normalize_answer, reader_from_contracts, token_f1

root = Path(__file__).resolve().parents[1]
models = load_models("cuda:0")
PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b|\b(?:1|2)\d{3}\b")


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


def joint(question, answer, evs):
    if not evs:
        return 0.0
    packed = Contract("", evs, "readout", "read", 0.5, 0, LIVE, "none", False)
    pred, _, _ = reader_from_contracts(models, question, [packed])
    return token_f1(pred, answer)


def bridge(question, answer, evs):
    solos = []
    for ev in evs:
        context = f"{ev.get('title', '')}: {ev.get('text', '')}".strip()
        span, score = models.answer(question, context)
        if not span or not PROPER.search(span):
            continue
        solos.append((score, normalize_answer(span), span, ev))
    groups = {}
    for score, needle, span, ev in solos:
        if len(needle) < 3:
            continue
        groups.setdefault(needle, []).append((score, span, ev))
    ranked = []
    for needle, items in groups.items():
        titles = set()
        holders = []
        for score, span, ev in items:
            titles.add(ev.get("title") or "")
            holders.append(ev)
        mentioned = [ev for ev in evs if needle in normalize_answer(f"{ev.get('title', '')} {ev.get('text', '')}")]
        page_titles = {ev.get("title") or "" for ev in mentioned}
        if len(page_titles) < 2:
            continue
        ranked.append((len(page_titles), max(score for score, _, _ in items), mentioned))
    if not ranked:
        return joint(question, answer, evs)
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return joint(question, answer, ranked[0][2][:6])


for dataset in ("hotpot", "2wiki"):
    with (root / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    print("\n==", dataset, flush=True)
    for setting in ("mixed", "unsupported"):
        for method in ("crag_plus", "trustmem"):
            left, right = [], []
            for index in range(len(meta)):
                question = meta[index]["question"]
                answer = meta[index]["answer"]
                evs = rows(packets[setting][method][index])
                left.append(joint(question, answer, evs))
                right.append(bridge(question, answer, evs))
            n = len(left)
            print(
                f"  {setting:12s} {method:10s} joint={sum(left)/n:.3f} bridge={sum(right)/n:.3f}",
                flush=True,
            )
