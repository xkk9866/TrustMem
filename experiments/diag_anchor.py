"""Gold coverage if later sentences that name a question entity are kept."""

import pickle
import re
from collections import defaultdict
from pathlib import Path

from run_trustmem import gold_recall, tokens

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b|\b(?:1|2)\d{3}\b")


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


def overlap(question, ev):
    q = tokens(question)
    if not q:
        return 0.0
    return len(q & tokens(f"{ev.get('title', '')} {ev.get('text', '')}")) / len(q)


def question_anchors(question):
    anchors = set()
    for name in NAME.findall(question):
        if len(name) >= 4:
            anchors.add(name.lower())
    return anchors


def select(question, evs, skip_lede: bool):
    anchors = question_anchors(question)
    ranked = sorted(evs, key=lambda ev: overlap(question, ev), reverse=True)
    hop = ranked[:2]
    hop_titles = {(ev.get("title") or "") for ev in hop}
    # First stored sentence of each page is treated as the lede.
    seen_title = set()
    lede = set()
    for ev in evs:
        title = ev.get("title") or ""
        if title not in seen_title:
            seen_title.add(title)
            lede.add(ev.get("text"))
    linked = []
    for ev in evs:
        if ev in hop:
            continue
        if skip_lede and ev.get("text") in lede:
            continue
        blob = f"{ev.get('title', '')} {ev.get('text', '')}".lower()
        if any(anchor in blob for anchor in anchors):
            linked.append(ev)
    linked.sort(key=lambda ev: overlap(question, ev))  # low overlap first: the bridge, not the bio
    # Prefer a page we do not already have.
    other = [ev for ev in linked if (ev.get("title") or "") not in hop_titles]
    pool = other or linked
    return hop + pool[:2]


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "clean"):
            stat = defaultdict(float)
            n = len(meta)
            for index, item in enumerate(meta):
                gold = item["answer"]
                question = item["question"]
                mem = rows(packets[setting]["trustmem"][index])
                plus = rows(packets[setting]["crag_plus"][index])
                stat["plus"] += gold_recall(" ".join(ev["text"] for ev in plus), gold) >= 0.99
                for key, skip in (("keep", False), ("skip", True)):
                    cut = select(question, mem, skip)
                    stat[key] += gold_recall(" ".join(ev["text"] for ev in cut), gold) >= 0.99
                    stat[key + "_n"] += len(cut)
            print(
                f"  {setting:12s} plus {stat['plus']/n:.3f}"
                f"  anchor {stat['keep']/n:.3f} ({stat['keep_n']/n:.1f})"
                f"  skip-lede {stat['skip']/n:.3f} ({stat['skip_n']/n:.1f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
