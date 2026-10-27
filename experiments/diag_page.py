"""Top two sentences on each cited page: does the answer stay?"""
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from run_trustmem import gold_recall

root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_cite60.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
scores = json.loads((root / "results" / "scores_cite60.json").read_text(encoding="utf-8"))


def per_page(question, stored, per=2):
    groups = {}
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            if not text:
                continue
            groups.setdefault(ev.get("title") or text[:40], []).append(text)
    chosen = []
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    for sents in groups.values():
        uniq = list(dict.fromkeys(sents))
        docs = [re.findall(r"[a-z0-9]+", text.lower()) or ["empty"] for text in uniq]
        ranker = BM25Okapi(docs)
        order = sorted(range(len(uniq)), key=lambda i: float(ranker.get_scores(query)[i]), reverse=True)
        chosen.extend(uniq[i] for i in order[:per])
    return chosen


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    gold = n = 0.0
    for index, stored in enumerate(packets[setting]["trustmem"]):
        kept = per_page(meta[index]["question"], stored, per=2)
        n += len(kept)
        gold += gold_recall(" ".join(kept), meta[index]["answer"])
    plus = scores[f"{setting}:crag_plus"]["f1"]
    tm = scores[f"{setting}:trustmem"]["f1"]
    print(f"{setting:12s} page2 n={n/60:.1f} gold={gold/60:.3f} current_f1 tm={tm:.3f} plus={plus:.3f}")
