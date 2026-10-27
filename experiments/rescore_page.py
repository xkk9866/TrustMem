"""Cited-page commitment.

CRAG+ keeps four BM25 sentences from the pages it cited. This readout scores
every sentence of those same pages alone, down-weights a span that only repeats
the page title, and optionally keeps a sentence only when it shares a content
word with the other cited page. Same frozen reader, same cited pages.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_durable import load_slices
from run_trustmem import load_models, normalize_answer, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")
CACHE = {}
STOP = tokens(
    "a an the of and in on for to was were is are be been by with from as at or "
    "his her their its this that which who he she it they them their"
)


def pages_of(item):
    out = {}
    for doc in item["docs"]:
        out.setdefault(doc["title"], [])
        for text in doc["sentences"]:
            if text not in out[doc["title"]]:
                out[doc["title"]].append(text)
    return out


def titles_of(stored):
    found = []
    for contract in stored:
        for ev in contract.evidence or []:
            title = ev.get("title") or ""
            if title and title not in found:
                found.append(title)
    return found


def read(question, title, text):
    key = (question, title, text)
    if key in CACHE:
        return CACHE[key]
    pred, score = MODELS.answer(question, f"{title}: {text}")
    CACHE[key] = (pred, float(score))
    return CACHE[key]


def penalized(question, title, text, titles):
    pred, conf = read(question, title, text)
    needle = normalize_answer(pred)
    if needle and any(needle == normalize_answer(name.split("(")[0]) or (len(needle) > 3 and needle in normalize_answer(name)) for name in titles):
        conf *= 0.25
    return pred, conf


def best(question, cands, titles):
    chosen = (0.0, "")
    for title, text in cands:
        pred, conf = penalized(question, title, text, titles)
        if conf > chosen[0]:
            chosen = (conf, pred)
    return chosen[1]


def bridge_cands(question, cited, page_map):
    blobs = {title: normalize_answer(" ".join(page_map.get(title, []))) for title in cited}
    q = tokens(question)
    kept = []
    for title in cited:
        others = " ".join(blobs[other] for other in cited if other != title)
        other_toks = tokens(others) - q - STOP
        for text in page_map.get(title, []):
            if tokens(text) - q - STOP & other_toks:
                kept.append((title, text))
    return kept


def packet_cands(stored):
    cands, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            title = ev.get("title") or ""
            if text and (title, text) not in seen:
                seen.add((title, text))
                cands.append((title, text))
    return cands


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
        for setting in ("mixed", "unsupported", "incomplete", "clean", "conflict"):
            scores = {name: [] for name in ("plus", "page", "bridge")}
            empty_bridge = 0
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                page_map = pages_of(by_q[question])
                cited = [title for title in titles_of(packets[setting]["crag_plus"][index]) if title in page_map]
                plus = packet_cands(packets[setting]["crag_plus"][index])
                page = [(title, text) for title in cited for text in page_map[title]]
                bridged = bridge_cands(question, cited, page_map)
                if not bridged:
                    empty_bridge += 1
                    bridged = page
                scores["plus"].append(token_f1(best(question, plus, cited), gold))
                scores["page"].append(token_f1(best(question, page, cited), gold))
                scores["bridge"].append(token_f1(best(question, bridged, cited), gold))
            print(
                f"  {setting:12s} plus {np.mean(scores['plus']):.3f}"
                f"  page {np.mean(scores['page']):.3f}"
                f"  bridge {np.mean(scores['bridge']):.3f}"
                f"  no-bridge {empty_bridge}",
                flush=True,
            )
            for name in ("page", "bridge"):
                delta, lo, hi = bootstrap(scores[name], scores["plus"])
                print(f"    {name}-plus: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
