"""Two measurements.

Parenthetical titles are not answers: a span that only repeats a name in
parentheses is down-weighted, then the same reader commits to one sentence.

Comparison questions name two candidates. Each specialist returns a witnessed
value (a release year, or the birth and death of the person a page names).
The shared memory stores the candidate that wins the comparison, instead of
a sentence the reader happens to copy.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.35, 0)

from diag_compare2 import pick
from run_durable import load_slices
from run_trustmem import load_models, normalize_answer, token_f1

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")
CACHE = {}
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
RANGE = re.compile(r"(1[0-9]{3})\s*[–—-]\s*(1[0-9]{3}|20[0-9]{2})")
DIRECTOR = re.compile(r"directed by ([A-Z][^,.(]{2,60})")


def packet(stored):
    rows, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            key = (ev.get("title") or "", ev.get("text") or "")
            if key[1] and key not in seen:
                seen.add(key)
                rows.append(key)
    return rows


def choices_of(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [match.group(1).strip(), match.group(2).strip()]


def solo(question, title, text):
    key = (question, title, text)
    if key not in CACHE:
        pred, score = MODELS.answer(question, f"{title}: {text}")
        CACHE[key] = (pred, float(score))
    return CACHE[key]


def factor(pred, titles, paren: bool):
    needle = normalize_answer(pred)
    if not needle or len(needle) <= 3:
        return 1.0
    for name in titles:
        head = normalize_answer(name.split("(")[0])
        if needle == head or needle in head:
            return 0.25
        if not paren:
            continue
        match = re.search(r"\(([^)]*)\)", name)
        if match and needle in normalize_answer(match.group(1)) and needle not in head:
            return 0.25
    return 1.0


def commit(question, rows, titles, paren: bool):
    best = (0.0, "")
    for title, text in rows:
        pred, conf = solo(question, title, text)
        conf *= factor(pred, titles, paren)
        if conf > best[0]:
            best = (conf, pred)
    return best[1]


def find_doc(docs, name):
    needle = normalize_answer(name)
    if not needle:
        return None
    for doc in docs:
        title = normalize_answer(doc["title"])
        if title == needle or title.startswith(needle) or needle.startswith(title):
            return doc
    return None


def doc_for_choice(docs, choice):
    return find_doc(docs, choice)


def lifespan(doc):
    if not doc:
        return None
    for sent in doc["sentences"][:4]:
        match = RANGE.search(sent)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def release_year(doc):
    if not doc:
        return None
    for sent in doc["sentences"][:2]:
        if re.search(r"film|album|song|novel|series", sent, re.I):
            match = YEAR.search(sent)
            if match:
                return int(match.group(1))
    if doc["sentences"]:
        match = YEAR.search(doc["sentences"][0])
        if match:
            return int(match.group(1))
    return None


def director_doc(docs, film):
    if not film:
        return None
    for sent in film["sentences"][:3]:
        match = DIRECTOR.search(sent)
        if match:
            return find_doc(docs, match.group(1).strip())
    return None


def compare(question, docs):
    opts = choices_of(question)
    if len(opts) != 2:
        return ""
    q = question.lower()
    pages = [doc_for_choice(docs, opt) for opt in opts]
    if "director" in q and any(word in q for word in ("died", "born", "older", "younger")):
        pages = [director_doc(docs, page) for page in pages]
        values = [lifespan(page) for page in pages]
        if any(value is None for value in values):
            return ""
        if "died" in q or "older" not in q and "born" not in q and "younger" not in q:
            key = lambda pair: pair[1]
            want_min = "later" not in q and "younger" not in q
        elif "born" in q or "older" in q or "younger" in q:
            key = lambda pair: pair[0]
            want_min = "later" not in q and "younger" not in q
        else:
            return ""
    elif "lived longer" in q or "live longer" in q:
        values = [lifespan(page) for page in pages]
        if any(value is None for value in values):
            return ""
        ages = [death - birth for birth, death in values]
        return opts[int(ages[1] > ages[0])]
    elif any(word in q for word in ("died", "born")):
        values = [lifespan(page) for page in pages]
        if any(value is None for value in values):
            return ""
        index = 1 if "died" in q else 0
        key = lambda pair: pair[index]
        want_min = "later" not in q and "after" not in q
    else:
        values = [release_year(page) for page in pages]
        if any(value is None for value in values):
            return ""
        key = lambda value: value
        want_min = not any(word in q for word in ("more recently", "later", "after"))
    if "director" in q or any(word in q for word in ("died", "born")):
        picked = min(range(2), key=lambda i: key(values[i])) if want_min else max(range(2), key=lambda i: key(values[i]))
        return opts[picked]
    picked = min(range(2), key=lambda i: values[i]) if want_min else max(range(2), key=lambda i: values[i])
    return opts[picked]


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
        for setting in ("mixed", "unsupported", "incomplete", "clean"):
            scores = {name: [] for name in ("head", "page", "judge", "both")}
            fired = 0
            for index, item in enumerate(meta):
                question, gold = item["question"], item["answer"]
                rows = packet(packets[setting]["crag_plus"][index])
                titles = list(dict.fromkeys(title for title, _ in rows))
                docs = by_q[question]["docs"]
                page_map = {doc["title"]: doc["sentences"] for doc in docs}
                page_rows = [(title, text) for title in titles for text in page_map.get(title, [])]
                head = commit(question, rows, titles, paren=False)
                page = commit(question, page_rows or rows, titles, paren=False)
                judged = pick(question, docs)
                if judged:
                    fired += 1
                scores["head"].append(token_f1(head, gold))
                scores["page"].append(token_f1(page, gold))
                scores["judge"].append(token_f1(judged or head, gold))
                scores["both"].append(token_f1(judged or page, gold))
            print(
                f"  {setting:12s} head {np.mean(scores['head']):.3f}"
                f"  page {np.mean(scores['page']):.3f}"
                f"  judge {np.mean(scores['judge']):.3f}"
                f"  both {np.mean(scores['both']):.3f}"
                f"  fired {fired}",
                flush=True,
            )
            for name in ("page", "judge", "both"):
                delta, lo, hi = bootstrap(scores[name], scores["head"])
                mark = " *" if lo > 0 else ""
                print(f"    {name}-head {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}", flush=True)


if __name__ == "__main__":
    main()
