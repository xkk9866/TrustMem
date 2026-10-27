"""Pick the answer span that occurs in only one stored sentence.

Bridge answers are locally unique. A biography repeats its subject, so that
span loses. A question that names two stored pages is read from those pages'
identity sentences instead, because the answer is one of the two names.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_trustmem import Contract, load_models, normalize_answer, reader_from_contracts, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")
YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")


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


def answer_joint(question, evs):
    if not evs:
        return ""
    packed = Contract("", evs, "readout", "read", 0.5, 0, 10**9, "none", False)
    pred, _, _ = reader_from_contracts(MODELS, question, [packed])
    return pred


def is_identity(ev):
    title = (ev.get("title") or "").split("(")[0].strip()
    text = ev.get("text") or ""
    if title and text.lower().startswith(title.lower()[: max(8, min(len(title), 24))].strip()):
        return True
    return "(" in text[:180] and YEAR.search(text[:180]) is not None


def titles_in_question(question, evs):
    found = []
    q = question.lower()
    seen = set()
    for ev in evs:
        title = (ev.get("title") or "").split("(")[0].strip()
        key = title.lower()
        if len(key) > 3 and key in q and key not in seen:
            seen.add(key)
            found.append(title)
    return found


def unique_span(question, evs):
    qnorm = normalize_answer(question)
    texts = [normalize_answer(ev.get("text") or "") for ev in evs]
    title_norms = {normalize_answer((ev.get("title") or "").split("(")[0]) for ev in evs}
    best = None
    for ev in evs:
        line = f"{ev.get('title', '')}: {ev.get('text', '')}".strip()
        pred, conf = MODELS.answer(question, line)
        needle = normalize_answer(pred)
        if not needle or conf < 0.01 or needle in qnorm or needle in title_norms:
            continue
        support = sum(needle in text for text in texts)
        if support != 1:
            continue
        if best is None or conf > best[0]:
            best = (conf, pred)
    return None if best is None else best[1]


def struct_answer(question, evs):
    named = titles_in_question(question, evs)
    if len(named) >= 2:
        wanted = {title.lower() for title in named}
        ident = []
        for ev in evs:
            title = (ev.get("title") or "").split("(")[0].strip().lower()
            if title in wanted and is_identity(ev):
                ident.append(ev)
        if ident:
            return answer_joint(question, ident), "compare"
        same = [ev for ev in evs if (ev.get("title") or "").split("(")[0].strip().lower() in wanted]
        if same:
            return answer_joint(question, same[:4]), "compare"
    span = unique_span(question, evs)
    if span:
        return span, "unique"
    q = tokens(question)
    ranked = sorted(evs, key=lambda ev: len(q & tokens(ev.get("text") or "")), reverse=True)
    return answer_joint(question, ranked[:4]), "joint"


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
        for setting in ("mixed", "unsupported", "incomplete", "clean", "conflict"):
            scores = {name: [] for name in ("crag_plus", "trustmem", "struct")}
            kinds = {"compare": 0, "unique": 0, "joint": 0}
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                plus = rows(packets[setting]["crag_plus"][index])
                mem = rows(packets[setting]["trustmem"][index])
                q = tokens(question)
                plus4 = sorted(plus, key=lambda ev: len(q & tokens(ev.get("text") or "")), reverse=True)[:4]
                mem4 = sorted(mem, key=lambda ev: len(q & tokens(ev.get("text") or "")), reverse=True)[:4]
                pred, kind = struct_answer(question, mem)
                kinds[kind] += 1
                scores["crag_plus"].append(token_f1(answer_joint(question, plus4), gold))
                scores["trustmem"].append(token_f1(answer_joint(question, mem4), gold))
                scores["struct"].append(token_f1(pred, gold))
            print(
                f"  {setting:12s} plus {np.mean(scores['crag_plus']):.3f}"
                f"  mem {np.mean(scores['trustmem']):.3f}"
                f"  struct {np.mean(scores['struct']):.3f}"
                f"  {kinds}",
                flush=True,
            )
            for base in ("crag_plus", "trustmem"):
                delta, lo, hi = bootstrap(scores["struct"], scores[base])
                print(f"    struct-{base}: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
