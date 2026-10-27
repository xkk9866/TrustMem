"""Where the long-horizon read loses the original sentence."""
import pickle
from pathlib import Path

import numpy as np

from run_durable import applicability, readout
from run_trustmem import gold_recall, load_models, tokens

root = Path(__file__).resolve().parents[1]
models = load_models("cuda:1")
with (root / "results" / "packets_hotpot.pkl").open("rb") as handle:
    test_packets, pool_packets, meta = pickle.load(handle)

setting = "mixed"
for method in ("crag_plus", "trustmem", "crag"):
    own_rank_ok = 0
    gold_in_top = 0
    gold_in_own = 0
    n = min(40, len(meta))
    interference = pool_packets["clean"][method]
    for index, stored in enumerate(test_packets[setting][method][:n]):
        answer = meta[index]["answer"]
        question = meta[index]["question"]
        own_blob = " ".join(c.evidence_text() for c in stored)
        if gold_recall(own_blob, answer) > 0:
            gold_in_own += 1
        start = index % len(interference)
        ordered = interference[start:] + interference[:start]
        mem = list(stored)
        for packet in ordered[:1000]:
            mem.extend(packet)
        qvec = models.embed(question, query=True)
        q_toks = tokens(question)
        scored = []
        seen = set()
        for contract in mem:
            for ev in contract.evidence or []:
                text = ev.get("text", "")
                if not text or text in seen:
                    continue
                seen.add(text)
                overlap = (len(q_toks & tokens(text)) / len(q_toks)) if q_toks else 0.0
                vec = models.embed(text, query=False)
                cos = float(np.dot(qvec, vec))
                scored.append((0.45 * overlap + 0.55 * max(cos, 0.0), text, contract.query_id == meta[index]["id"]))
        scored.sort(reverse=True)
        top = scored[:4]
        if any(item[2] for item in top):
            own_rank_ok += 1
        blob = " ".join(item[1] for item in top)
        if gold_recall(blob, answer) > 0:
            gold_in_top += 1
    print(method, "gold_in_own", gold_in_own, "own_in_top4", own_rank_ok, "gold_in_top4", gold_in_top, "n", n)
