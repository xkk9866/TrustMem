"""Overwrite a near-copy in the attested log with the page-faithful sentence.

The corrective channel is the same page read CRAG+ uses. A sentence in the
TrustMem log is replaced only when it is a near-copy of that read and not the
same text. Every other attested sentence stays. The reader is unchanged.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_trustmem import Contract, load_models, reader_from_contracts, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")
TAU = 0.62


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


def rank_top(question, evs, k=4):
    if not evs:
        return []
    q_toks = tokens(question)
    qvec = MODELS.embed(question, query=True)
    scored = []
    for ev in evs:
        text = ev.get("text") or ""
        overlap = (len(q_toks & tokens(text)) / len(q_toks)) if q_toks else 0.0
        cos = float(np.dot(qvec, MODELS.embed(text, query=False)))
        scored.append((0.45 * overlap + 0.55 * max(cos, 0.0), ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [ev for _, ev in scored[:k]]


def answer(question, evs):
    if not evs:
        return ""
    packed = Contract(
        claim="",
        evidence=evs,
        producer="readout",
        query_id="read",
        confidence=0.5,
        t_start=0,
        t_end=10**9,
        fault="none",
        injected=False,
    )
    pred, _, _ = reader_from_contracts(MODELS, question, [packed])
    return pred


def faithful(tm_evs, plus_evs, question):
    """Keep the attested ranking. Swap a top sentence for its page-faithful near-copy."""
    ranked = rank_top(question, tm_evs, k=4)
    swapped = []
    n_swap = 0
    used = set()
    for ev in ranked:
        best = None
        best_f1 = 0.0
        for other in plus_evs:
            if other.get("text") == ev.get("text"):
                best = None
                best_f1 = 1.0
                break
            score = token_f1(ev.get("text") or "", other.get("text") or "")
            if score > best_f1:
                best_f1 = score
                best = other
        if best is not None and best_f1 >= TAU and best.get("text") not in used:
            swapped.append(best)
            used.add(best.get("text"))
            n_swap += 1
        else:
            swapped.append(ev)
            used.add(ev.get("text"))
    return swapped, n_swap


def residual(tm_evs, plus_evs, question):
    """Page-faithful top sentences, plus attested sentences that are not near-copies."""
    base = rank_top(question, plus_evs, k=4)
    have = {ev.get("text") for ev in base}
    extra = []
    for ev in rank_top(question, tm_evs, k=8):
        text = ev.get("text") or ""
        if text in have:
            continue
        near = max((token_f1(text, other.get("text") or "") for other in plus_evs), default=0.0)
        if near >= TAU:
            continue
        extra.append(ev)
        if len(extra) >= 2:
            break
    return base + extra


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
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
        for setting in ("mixed", "unsupported", "incomplete", "conflict", "clean", "paraphrase"):
            if setting not in packets:
                continue
            scores = {name: [] for name in ("crag_plus", "trustmem", "faithful", "residual")}
            swaps = 0
            for index, question_meta in enumerate(meta):
                question = question_meta["question"]
                gold = question_meta["answer"]
                plus = rows(packets[setting]["crag_plus"][index])
                mem = rows(packets[setting]["trustmem"][index])
                pred_plus = answer(question, rank_top(question, plus))
                pred_mem = answer(question, rank_top(question, mem))
                fixed, n_swap = faithful(mem, plus, question)
                swaps += int(n_swap > 0)
                pred_fixed = answer(question, fixed)
                pred_res = answer(question, residual(mem, plus, question))
                scores["crag_plus"].append(token_f1(pred_plus, gold))
                scores["trustmem"].append(token_f1(pred_mem, gold))
                scores["faithful"].append(token_f1(pred_fixed, gold))
                scores["residual"].append(token_f1(pred_res, gold))
            print(
                f"  {setting:12s} plus {np.mean(scores['crag_plus']):.3f}"
                f"  mem {np.mean(scores['trustmem']):.3f}"
                f"  faithful {np.mean(scores['faithful']):.3f}"
                f"  residual {np.mean(scores['residual']):.3f}"
                f"  swapped {swaps}/{len(meta)}",
                flush=True,
            )
            for name in ("faithful", "residual"):
                for base in ("crag_plus", "trustmem"):
                    delta, lo, hi = bootstrap(scores[name], scores[base])
                    print(f"    {name}-{base}: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
