"""Counterfactual attestation.

A sentence from one specialist is kept only when adding it changes the other
specialist's answer and the new answer is a span of that sentence. Shared words
are not a witness. The same frozen reader scores every method.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_trustmem import Contract, load_models, normalize_answer, reader_from_contracts, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")
CACHE = {}


def rows(stored):
    grouped = {}
    order = []
    for contract in stored:
        producer = contract.producer or "agent"
        if producer not in grouped:
            grouped[producer] = []
            order.append(producer)
        seen = {ev.get("text") for ev in grouped[producer]}
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                grouped[producer].append(ev)
                seen.add(text)
    return grouped, order


def flat(grouped):
    out, seen = [], set()
    for evs in grouped.values():
        for ev in evs:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


def read(question, evs):
    if not evs:
        return "", 0.0
    key_ev = tuple((ev.get("title", ""), ev.get("text", "")) for ev in evs)
    key = (question, key_ev)
    if key in CACHE:
        return CACHE[key]
    packed = Contract("", list(evs), "readout", "read", 0.5, 0, 10**9, "none", False)
    pred, score, _ = reader_from_contracts(MODELS, question, [packed])
    CACHE[key] = (pred, float(score))
    return CACHE[key]


def focus(question, evs, k=4):
    q = tokens(question)
    ranked = sorted(evs, key=lambda ev: len(q & tokens(f"{ev.get('title', '')} {ev.get('text', '')}")), reverse=True)
    return ranked[:k]


def commit(question, evs):
    titles = {normalize_answer((ev.get("title") or "").split("(")[0]) for ev in evs}
    best = (0.0, "")
    for ev in evs:
        pred, conf = read(question, [ev])
        needle = normalize_answer(pred)
        if needle and any(needle == title or (len(needle) > 3 and needle in title) for title in titles if title):
            conf *= 0.25
        if conf > best[0]:
            best = (conf, pred)
    return best[1]


def causal(question, grouped, order):
    """Return the answer moved by the other specialist, else the strongest solo span."""
    if len(order) < 2:
        return commit(question, flat(grouped)), False
    proposals = []
    for agent in order:
        others = []
        for other in order:
            if other != agent:
                others.extend(focus(question, grouped[other], 4))
        base, _ = read(question, others)
        base_norm = normalize_answer(base)
        for ev in grouped[agent]:
            pred, conf = read(question, others + [ev])
            needle = normalize_answer(pred)
            if not needle or needle == base_norm:
                continue
            if needle not in normalize_answer(ev.get("text") or ""):
                continue
            proposals.append((conf, pred))
    if not proposals:
        return commit(question, flat(grouped)), False
    return max(proposals)[1], True


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if len(diff) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "clean", "conflict"):
            if setting not in packets:
                continue
            scores = {name: [] for name in ("crag_joint", "crag_commit", "causal")}
            fired = 0
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                grouped, order = rows(packets[setting]["trustmem"][index])
                plus = flat(rows(packets[setting]["crag_plus"][index])[0])
                pred_c, did = causal(question, grouped, order)
                fired += int(did)
                scores["crag_joint"].append(token_f1(read(question, plus)[0], gold))
                scores["crag_commit"].append(token_f1(commit(question, plus), gold))
                scores["causal"].append(token_f1(pred_c, gold))
            print(
                f"  {setting:12s} joint {np.mean(scores['crag_joint']):.3f}"
                f"  commit {np.mean(scores['crag_commit']):.3f}"
                f"  causal {np.mean(scores['causal']):.3f}"
                f"  fired {fired}/{len(meta)}",
                flush=True,
            )
            for base in ("crag_joint", "crag_commit"):
                delta, lo, hi = bootstrap(scores["causal"], scores[base])
                print(f"    causal-{base}: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
