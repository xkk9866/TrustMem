"""Same pages, two uses.

The comparison reads a release year or a life span from each candidate page
and keeps the candidate that wins. The control gives the frozen reader every
sentence of those same pages and lets it answer the original question.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.20, 0)

from diag_compare2 import pick, support_pages
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
    for dataset in ("hotpot", "2wiki"):
        _, test, _ = load_slices(dataset, 120, 4)
        by_q = {item["question"]: item for item in test}
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        settings = ("mixed",) if dataset == "2wiki" else ("mixed", "unsupported", "incomplete", "clean")
        for setting in settings:
            head_s, judge_s, reader_s = [], [], []
            fired = 0
            for index, item in enumerate(meta):
                question, gold = item["question"], item["answer"]
                docs = by_q[question]["docs"]
                rows = packet(packets[setting]["crag_plus"][index])
                head = commit(question, rows)
                judged = pick(question, docs)
                if judged:
                    fired += 1
                    witness = []
                    for doc in support_pages(question, docs):
                        for text in doc["sentences"]:
                            witness.append((doc["title"], text))
                    reader = commit(question, witness) if witness else head
                else:
                    reader = head
                head_s.append(token_f1(head, gold))
                judge_s.append(token_f1(judged or head, gold))
                reader_s.append(token_f1(reader if judged else head, gold))
            print(
                f"  {setting:12s} head {np.mean(head_s):.3f}"
                f"  judge {np.mean(judge_s):.3f}"
                f"  same-pages {np.mean(reader_s):.3f}"
                f"  fired {fired}",
                flush=True,
            )
            for name, scores in (("judge", judge_s), ("same-pages", reader_s)):
                delta, lo, hi = bootstrap(scores, head_s)
                mark = " *" if lo > 0 else ""
                print(f"    {name}-head {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}", flush=True)
            if fired:
                delta, lo, hi = bootstrap(judge_s, reader_s)
                mark = " *" if lo > 0 else ""
                print(f"    judge-same-pages {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}", flush=True)


if __name__ == "__main__":
    main()
