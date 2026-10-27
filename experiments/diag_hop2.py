"""Gold coverage of a two-hop cut: question sentence, then the sentence that names its new entity."""
import pickle
import re
from pathlib import Path

from run_trustmem import gold_recall, tokens

root = Path(__file__).resolve().parents[1]
NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")


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


def hop(question, evs):
    if not evs:
        return []
    q = tokens(question)
    def overlap(ev):
        return len(q & tokens(f"{ev.get('title', '')} {ev.get('text', '')}"))
    hop1 = max(evs, key=overlap)
    fresh = []
    for name in NAME.findall(f"{hop1.get('title', '')} {hop1.get('text', '')}"):
        if tokens(name) - q:
            fresh.append(name)
    hop2 = None
    best = -1
    for ev in evs:
        if ev["text"] == hop1["text"]:
            continue
        text = ev.get("text") or ""
        hits = sum(1 for name in fresh if name in text or name in (ev.get("title") or ""))
        if hits > best:
            best = hits
            hop2 = ev
    chosen = [hop1]
    if hop2 is not None and best > 0:
        chosen.append(hop2)
    return chosen


for dataset in ("hotpot", "2wiki"):
    with (root / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    print("\n==", dataset)
    for setting in ("mixed", "unsupported", "incomplete"):
        for method in ("crag_plus", "trustmem"):
            gold = n = 0.0
            count = 0
            for index, stored in enumerate(packets[setting][method]):
                evs = rows(stored)
                count += 1
                if not evs:
                    continue
                kept = hop(meta[index]["question"], evs)
                n += len(kept)
                gold += gold_recall(" ".join(ev["text"] for ev in kept), meta[index]["answer"])
            print(f"  {setting:12s} {method:10s} n={n/count:.1f} gold={gold/count:.3f}")
