"""Print unsupported packets where the answer is stored but not in the BM25 top."""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from run_trustmem import gold_recall, tokens

root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_pilot.pkl").open("rb") as handle:
    test_packets, _, meta = pickle.load(handle)


def score_map(question, sents):
    docs = [re.findall(r"[a-z0-9]+", text.lower()) or ["empty"] for text in sents]
    ranker = BM25Okapi(docs)
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    raw = ranker.get_scores(query)
    return [float(x) for x in raw]


shown = 0
for index, stored in enumerate(test_packets["unsupported"]["trustmem"]):
    rows = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                rows.append((contract.producer, ev.get("title", ""), text))
    if not rows:
        continue
    answer = meta[index]["answer"]
    question = meta[index]["question"]
    texts = [text for _, _, text in rows]
    if gold_recall(" ".join(texts), answer) < 0.99:
        continue
    scores = score_map(question, texts)
    order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    top = " ".join(texts[i] for i in order[:4])
    if gold_recall(top, answer) > 0.2:
        continue
    print("\nQ:", question)
    print("A:", answer)
    for rank, i in enumerate(order):
        producer, title, text = rows[i]
        mark = "GOLD" if gold_recall(text, answer) > 0.5 else "    "
        print(f"  {rank:02d} {mark} {scores[i]:6.2f} {producer:8s} {title[:28]:28s} {text[:160]}")
    shown += 1
    if shown >= 4:
        break
print("\nshown", shown)
