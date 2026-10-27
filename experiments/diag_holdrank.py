"""Do the witness sentences stay ahead of 1000 later writes under lexical rank?"""

import pickle
from pathlib import Path

from diag_compare2 import pick, support_pages
from run_trustmem import tokens

ROOT = Path(__file__).resolve().parents[1]


def packet_rows(stored):
    rows = []
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text:
                rows.append((ev.get("title") or "", text))
    return rows


def overlap(question, text):
    q = tokens(question)
    if not q:
        return 0.0
    return len(q & tokens(text)) / len(q)


def main():
    from run_durable import load_slices

    _, test, pool_items = load_slices("2wiki", 120, 1000)
    by_q = {item["question"]: item for item in test}
    with (ROOT / "results" / "packets_2wiki.pkl").open("rb") as handle:
        _, pool, meta = pickle.load(handle)
    later = []
    for stored in pool["clean"]["crag_plus"]:
        later.extend(packet_rows(stored))
    print("later sentences", len(later), flush=True)
    fired = top2 = 0
    for item in meta:
        question = item["question"]
        docs = by_q[question]["docs"]
        if not pick(question, docs):
            continue
        fired += 1
        witness = []
        for doc in support_pages(question, docs):
            if doc["sentences"]:
                witness.append((doc["title"], doc["sentences"][0]))
        ranked = sorted(witness + later, key=lambda row: overlap(question, f"{row[0]} {row[1]}"), reverse=True)
        keys = {(title, text) for title, text in witness}
        hit = sum(1 for row in ranked[:4] if row in keys)
        top2 += int(hit >= min(2, len(witness)) and hit > 0)
    print(f"fired {fired} witness-in-top4 {top2}", flush=True)


if __name__ == "__main__":
    main()
