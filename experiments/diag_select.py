"""CPU-only gold coverage of a bridge selector. Not a test score."""

import pickle
import re
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


def names_of(ev, question):
    q = tokens(question)
    found = set()
    title = (ev.get("title") or "").split("(")[0].strip()
    if title:
        found.add(title)
    found.update(NAME.findall(ev.get("text") or ""))
    kept = set()
    for name in found:
        name = name.strip()
        if len(name) < 4:
            continue
        if tokens(name) and tokens(name) <= q:
            continue
        kept.add(name.lower())
    return kept


def bridge_cut(question, evs):
    ranked = sorted(evs, key=lambda ev: overlap(question, ev), reverse=True)
    hop = ranked[:2]
    introduced = set()
    hop_titles = set()
    for ev in hop:
        introduced |= names_of(ev, question)
        hop_titles.add(ev.get("title") or "")
    linked = []
    for ev in evs:
        if ev in hop:
            continue
        if (ev.get("title") or "") in hop_titles:
            continue
        blob = f"{ev.get('title', '')} {ev.get('text', '')}".lower()
        hits = sum(name in blob for name in introduced)
        if hits:
            linked.append((hits, ev))
    linked.sort(key=lambda item: item[0], reverse=True)
    return hop + [ev for _, ev in linked[:2]]


def page_diverse(question, evs):
    by_title = {}
    for ev in evs:
        by_title.setdefault(ev.get("title") or ev["text"][:24], []).append(ev)
    pages = []
    for title, group in by_title.items():
        best = max(group, key=lambda ev: overlap(question, ev))
        pages.append((overlap(question, best), title, group, best))
    pages.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    for _, title, group, best in pages[:4]:
        # On each kept page, prefer the sentence that names another kept page.
        other_titles = [item[1] for item in pages[:4] if item[1] != title]
        def link_score(ev):
            blob = (ev.get("text") or "").lower()
            return sum(len(name) > 3 and name.lower() in blob for name in other_titles)
        linked = max(group, key=lambda ev: (link_score(ev), overlap(question, ev)))
        chosen.append(linked if link_score(linked) else best)
    return chosen


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "conflict"):
            stat = {key: 0.0 for key in ("mem", "plus", "bridge", "page")}
            n_bridge = n_page = 0
            n = len(meta)
            for index, item in enumerate(meta):
                gold = item["answer"]
                question = item["question"]
                mem = rows(packets[setting]["trustmem"][index])
                plus = rows(packets[setting]["crag_plus"][index])
                stat["mem"] += gold_recall(" ".join(ev["text"] for ev in mem), gold) >= 0.99
                stat["plus"] += gold_recall(" ".join(ev["text"] for ev in plus), gold) >= 0.99
                cut = bridge_cut(question, mem)
                page = page_diverse(question, mem)
                n_bridge += len(cut)
                n_page += len(page)
                stat["bridge"] += gold_recall(" ".join(ev["text"] for ev in cut), gold) >= 0.99
                stat["page"] += gold_recall(" ".join(ev["text"] for ev in page), gold) >= 0.99
            print(
                f"  {setting:12s} full {stat['mem']/n:.3f} plus {stat['plus']/n:.3f}"
                f"  bridge {stat['bridge']/n:.3f} ({n_bridge/n:.1f})"
                f"  page {stat['page']/n:.3f} ({n_page/n:.1f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
