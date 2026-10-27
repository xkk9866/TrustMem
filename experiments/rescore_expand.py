"""Where the answer sentence sits on a cited page.

Compares the BM25 packet with three expansions that a specialist can do
from the pages it already cited: the next sentence, a longer BM25 prefix,
and a sentence that names an entity from the other specialist's evidence.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi

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
        rows = []
        for text in doc["sentences"]:
            if text not in rows:
                rows.append(text)
        out[doc["title"]] = rows
    return out


def titles_of(stored):
    found = []
    for contract in stored:
        for ev in contract.evidence or []:
            title = ev.get("title") or ""
            if title and title not in found:
                found.append(title)
    return found


def packet_cands(stored):
    cands, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            key = (ev.get("title") or "", ev.get("text") or "")
            if key[1] and key not in seen:
                seen.add(key)
                cands.append(key)
    return cands


def read(question, title, text, titles):
    key = (question, title, text)
    if key not in CACHE:
        pred, score = MODELS.answer(question, f"{title}: {text}")
        CACHE[key] = (pred, float(score))
    pred, conf = CACHE[key]
    needle = normalize_answer(pred)
    if needle and any(
        needle == normalize_answer(name.split("(")[0]) or (len(needle) > 3 and needle in normalize_answer(name))
        for name in titles
    ):
        conf *= 0.25
    return pred, conf


def best(question, cands, titles):
    chosen = (0.0, "")
    for title, text in cands:
        pred, conf = read(question, title, text, titles)
        if conf > chosen[0]:
            chosen = (conf, pred)
    return chosen[1]


def rank_pages(question, cited, page_map):
    rows = [(title, i, text) for title in cited for i, text in enumerate(page_map.get(title, []))]
    if not rows:
        return []
    tokenized = [re.findall(r"[a-z0-9]+", text.lower()) for _, _, text in rows]
    ranker = BM25Okapi(tokenized)
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    order = np.argsort(-ranker.get_scores(query))
    return [rows[i] for i in order]


def content(text, question):
    return tokens(text) - tokens(question) - STOP


def expand_next(plus, page_map):
    extra = []
    seen = set(plus)
    for title, text in plus:
        sents = page_map.get(title, [])
        if text in sents:
            i = sents.index(text)
            if i + 1 < len(sents):
                key = (title, sents[i + 1])
                if key not in seen:
                    seen.add(key)
                    extra.append(key)
    return plus + extra


def expand_entity(plus, cited, page_map, question):
    extra, seen = [], set(plus)
    for title, text in plus:
        seeds = content(text, question)
        if not seeds:
            continue
        for other in cited:
            if other == title:
                continue
            for sent in page_map.get(other, []):
                if content(sent, question) & seeds:
                    key = (other, sent)
                    if key not in seen:
                        seen.add(key)
                        extra.append(key)
    return plus + extra


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    ks = (4, 8, 12)
    names = ("plus", "next", "entity", "k4", "k8", "k12", "all")
    for dataset in ("hotpot", "2wiki"):
        _, test, _ = load_slices(dataset, 120, 4)
        by_q = {item["question"]: item for item in test}
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "clean"):
            scores = {name: [] for name in names}
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                page_map = pages_of(by_q[question])
                cited = [title for title in titles_of(packets[setting]["crag_plus"][index]) if title in page_map]
                plus = [(title, text) for title, text in packet_cands(packets[setting]["crag_plus"][index]) if title in page_map]
                ranked = rank_pages(question, cited, page_map)
                page = [(title, text) for title, _, text in ranked]
                scores["plus"].append(token_f1(best(question, plus, cited), gold))
                scores["next"].append(token_f1(best(question, expand_next(plus, page_map), cited), gold))
                scores["entity"].append(token_f1(best(question, expand_entity(plus, cited, page_map, question), cited), gold))
                for k in ks:
                    scores[f"k{k}"].append(token_f1(best(question, page[:k], cited), gold))
                scores["all"].append(token_f1(best(question, page, cited), gold))
            base = scores["plus"]
            parts = [f"{name} {np.mean(scores[name]):.3f}" for name in names]
            print(f"  {setting:12s} " + "  ".join(parts), flush=True)
            for name in names:
                if name == "plus":
                    continue
                delta, lo, hi = bootstrap(scores[name], base)
                mark = " *" if lo > 0 else ""
                print(f"    {name}-plus {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}", flush=True)


if __name__ == "__main__":
    main()
