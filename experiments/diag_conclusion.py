"""A stored comparison names both candidates, so later writes rarely outrank it."""

import pickle
from pathlib import Path

from diag_compare2 import choices_of, pick
from diag_holdrank import overlap, packet_rows
from run_durable import load_slices

ROOT = Path(__file__).resolve().parents[1]


def main():
    _, test, _ = load_slices("2wiki", 120, 1000)
    by_q = {item["question"]: item for item in test}
    with (ROOT / "results" / "packets_2wiki.pkl").open("rb") as handle:
        _, pool, meta = pickle.load(handle)
    later = []
    for stored in pool["clean"]["crag_plus"]:
        later.extend(packet_rows(stored))
    top1 = fired = 0
    for item in meta:
        question = item["question"]
        docs = by_q[question]["docs"]
        answer = pick(question, docs)
        opts = choices_of(question)
        if not answer or len(opts) != 2:
            continue
        fired += 1
        text = f"Between {opts[0]} and {opts[1]}, the comparison selects {answer}."
        ranked = sorted([(text, overlap(question, text))] + [(row[1], overlap(question, f"{row[0]} {row[1]}")) for row in later], key=lambda pair: pair[1], reverse=True)
        top1 += int(ranked[0][0] == text)
    print(f"fired {fired} conclusion-rank1 {top1} later {len(later)}")


if __name__ == "__main__":
    main()
