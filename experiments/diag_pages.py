"""Keep a whole page when it shares a proper name with another page or its title is in the question."""
import pickle
import re
from pathlib import Path

from run_trustmem import gold_recall, tokens

root = Path(__file__).resolve().parents[1]
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


for dataset in ("hotpot", "2wiki"):
    with (root / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    print("\n==", dataset)
    for setting in ("mixed", "unsupported", "incomplete"):
        for method in ("crag_plus", "trustmem"):
            gold = n = full = 0.0
            count = 0
            for index, stored in enumerate(packets[setting][method]):
                evs = rows(stored)
                count += 1
                if not evs:
                    continue
                full += gold_recall(" ".join(ev["text"] for ev in evs), meta[index]["answer"])
                kept = keep_pages(meta[index]["question"], evs)
                n += len(kept)
                gold += gold_recall(" ".join(ev["text"] for ev in kept), meta[index]["answer"])
            print(
                f"  {setting:12s} {method:10s} n={n/count:.1f} gold={gold/count:.3f} full={full/count:.3f}"
            )
