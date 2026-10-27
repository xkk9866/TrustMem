"""Stored comparison after 1000 later writes.

The log holds the page-faithful packet, one sentence that names both
candidates and the selected value, and every later clean write. Rank is
lexical overlap with the question. If the stored conclusion is first, the
memory returns the value it already compared. Otherwise the frozen reader
commits to one of the top four sentences.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.18, 0)

from diag_compare2 import choices_of, pick
from diag_holdrank import overlap, packet_rows
from run_durable import load_slices
from run_trustmem import load_models, token_f1

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")


def commit(question, rows):
    best = (0.0, "")
    for title, text in rows:
        pred, conf = MODELS.answer(question, f"{title}: {text}")
        if conf > best[0]:
            best = (conf, pred)
    return best[1]


def packet(stored):
    rows, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            key = (ev.get("title") or "", ev.get("text") or "")
            if key[1] and key not in seen:
                seen.add(key)
                rows.append(key)
    return rows


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    _, test, _ = load_slices("2wiki", 120, 1000)
    by_q = {item["question"]: item for item in test}
    with (ROOT / "results" / "packets_2wiki.pkl").open("rb") as handle:
        packets, pool, meta = pickle.load(handle)
    later = []
    for stored in pool["clean"]["crag_plus"]:
        later.extend(packet_rows(stored))
    print("later", len(later), flush=True)
    for setting in ("mixed", "unsupported", "incomplete", "clean"):
        head_s, live_s = [], []
        first = 0
        for index, item in enumerate(meta):
            question, gold = item["question"], item["answer"]
            docs = by_q[question]["docs"]
            rows = packet(packets[setting]["crag_plus"][index])
            head_pred = commit(question, rows)
            head_s.append(token_f1(head_pred, gold))
            answer = pick(question, docs)
            opts = choices_of(question)
            extra = []
            if answer and len(opts) == 2:
                text = f"Between {opts[0]} and {opts[1]}, the comparison selects {answer}."
                extra.append(text)
            ranked = [(overlap(question, f"{title} {text}"), 0, "") for title, text in rows + later]
            for text in extra:
                ranked.append((overlap(question, text), 1, answer))
            ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
            if ranked and ranked[0][2]:
                first += 1
                pred = ranked[0][2]
            else:
                pred = head_pred
            live_s.append(token_f1(pred, gold))
        delta, lo, hi = bootstrap(live_s, head_s)
        mark = " *" if lo > 0 else ""
        print(
            f"  {setting:12s} head {np.mean(head_s):.3f}  live {np.mean(live_s):.3f}"
            f"  {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}  conclusion-first {first}",
            flush=True,
        )


if __name__ == "__main__":
    main()
