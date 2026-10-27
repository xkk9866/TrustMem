"""Prefer a sentence that names the other admitted page."""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from run_trustmem import gold_recall

root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_cite60.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)


def mention_first(question, stored, per=2):
    grouped = {}
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            title = ev.get("title") or ""
            if text:
                grouped.setdefault(title or text[:40], []).append(text)
    titles = [title for title in grouped if len(title) > 3]
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    chosen = []
    for title, sents in grouped.items():
        uniq = list(dict.fromkeys(sents))
        others = [other.lower() for other in titles if other != title]
        mentioned = [text for text in uniq if any(other in text.lower() for other in others)]
        if mentioned:
            chosen.extend(mentioned[:per])
            continue
        docs = [re.findall(r"[a-z0-9]+", text.lower()) or ["empty"] for text in uniq]
        ranker = BM25Okapi(docs)
        order = sorted(range(len(uniq)), key=lambda i: float(ranker.get_scores(query)[i]), reverse=True)
        chosen.extend(uniq[i] for i in order[:per])
    return chosen[:6]


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    gold = n = mention_items = 0.0
    for index, stored in enumerate(packets[setting]["trustmem"]):
        kept = mention_first(meta[index]["question"], stored)
        n += len(kept)
        gold += gold_recall(" ".join(kept), meta[index]["answer"])
    print(f"{setting:12s} n={n/60:.1f} gold={gold/60:.3f}")
