"""Witness algebra.

A multi-hop question is an expression over page-local witnesses. Comparison
selects the candidate whose witnessed value wins. Relational follow reads the
object of a relation on one page, then reads the asked attribute on the page
that object names. The control asks the original question of the same pages.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.22, 0)

from diag_compare2 import death_only, find_doc, pick
from run_durable import load_slices
from run_trustmem import load_models, normalize_answer, token_f1

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")
PERSON = r"([A-Z][A-Za-zÁÉÍÓÚÜÑáéíóúüñ.'’\-]+(?:\s+[A-Z][A-Za-zÁÉÍÓÚÜÑáéíóúüñ.'’\-]+){0,4})"


def packet(stored):
    rows, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            key = (ev.get("title") or "", ev.get("text") or "")
            if key[1] and key not in seen:
                seen.add(key)
                rows.append(key)
    return rows


def commit(question, rows):
    best = (0.0, "")
    titles = {normalize_answer(title.split("(")[0]) for title, _ in rows}
    for title, text in rows:
        pred, conf = MODELS.answer(question, f"{title}: {text}")
        needle = normalize_answer(pred)
        if needle and any(needle == name or (len(needle) > 3 and needle in name) for name in titles if name):
            conf *= 0.25
        if conf > best[0]:
            best = (conf, pred)
    return best[1]


def read(question, doc):
    if not doc or not doc["sentences"]:
        return ""
    context = " ".join(f"{doc['title']}: {sent}" for sent in doc["sentences"][:6])
    pred, _ = MODELS.answer(question, context)
    return pred


def anchor(question, docs):
    hits = []
    folded = question.lower()
    for doc in docs:
        title = doc["title"]
        if len(title) > 3 and title.lower() in folded:
            hits.append(doc)
    hits.sort(key=lambda doc: len(doc["title"]), reverse=True)
    return hits[0] if hits else None


def clean_person(name):
    name = re.split(r"\s+(?:and|who|with|from|for|starring|of the|in the)\s+", name.strip(" ,.;"))[0]
    name = re.sub(r"^(?:Sultan|King|Queen|Prince|Princess|Sir|Saint)\s+", "", name)
    return " ".join(name.split()[:5])


def grab(blob, pattern):
    match = re.search(pattern, blob)
    return clean_person(match.group(1)) if match else ""


def bridge(question, doc):
    blob = " ".join(doc["sentences"][:3]) if doc else ""
    q = question.lower()
    if "director" in q:
        return grab(blob, rf"directed by {PERSON}")
    if "composer" in q:
        return grab(blob, rf"(?:music by|composed by|score by) {PERSON}")
    if "performer" in q or "singer" in q:
        return grab(blob, rf"(?:performed by|sung by|recorded by|written and performed by) {PERSON}")
    if "founder" in q:
        found = grab(blob, rf"founded by {PERSON}") or grab(blob, rf"{PERSON}, co-founder")
        return found
    if "mother-in-law" in q or "father-in-law" in q:
        return grab(blob, rf"(?:wife|husband) of {PERSON}")
    if "father" in q or "mother" in q:
        return grab(blob, rf"(?:son|daughter) of {PERSON}")
    return ""


def attribute_question(question, person):
    q = question.lower()
    if "mother-in-law" in q:
        return f"Who is the mother of {person}?"
    if "father-in-law" in q:
        return f"Who is the father of {person}?"
    if any(word in q for word in ("born", "birth")):
        return f"Where was {person} born?"
    if any(word in q for word in ("die", "died", "death")):
        if q.startswith("when") or "when did" in q:
            return f"When did {person} die?"
        return f"Where did {person} die?"
    if "work" in q:
        return f"Where does {person} work?"
    if "country" in q or re.search(r"\bfrom\b", q):
        return f"What country is {person} from?"
    return ""


def asks_for_the_person(question):
    q = question.lower()
    if q.startswith(("where", "when", "which country", "what country", "what is the place", "what is the country")):
        return False
    if any(word in q for word in ("born", "birth", "die", "died", "death", "work", "country", "in-law")):
        return False
    return q.startswith("who")


def follow(question, docs):
    judged = pick(question, docs)
    if judged:
        return judged, "compare", []
    page = anchor(question, docs)
    if page is None:
        return "", "", []
    person = bridge(question, page)
    if not person:
        return "", "", []
    target = find_doc(docs, person)
    visited = [page] + ([target] if target is not None else [])
    if asks_for_the_person(question):
        return person, "bridge", visited
    local = attribute_question(question, target["title"] if target else person)
    if not local:
        return "", "", []
    if local.startswith("When did") and target is not None:
        year = death_only(target)
        if year:
            return str(year), "follow", visited
    if target is None:
        return "", "", []
    pred = read(local, target)
    if not pred or normalize_answer(pred) == normalize_answer(target["title"]):
        return "", "", []
    return pred, "follow", visited


def original_on_pages(question, pages):
    rows = []
    for doc in pages:
        for sent in (doc["sentences"][:4] if doc else []):
            rows.append((doc["title"], sent))
    return commit(question, rows) if rows else ""


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    for dataset in ("2wiki", "hotpot"):
        _, test, _ = load_slices(dataset, 120, 4)
        by_q = {item["question"]: item for item in test}
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "clean"):
            scores = {name: [] for name in ("head", "algebra", "same")}
            fired = {"compare": 0, "follow": 0, "bridge": 0}
            for index, item in enumerate(meta):
                question, gold = item["question"], item["answer"]
                docs = by_q[question]["docs"]
                rows = packet(packets[setting]["crag_plus"][index])
                head = commit(question, rows)
                pred, how, pages = follow(question, docs)
                if how in fired:
                    fired[how] += 1
                same = original_on_pages(question, pages) if how in {"follow", "bridge"} else head
                scores["head"].append(token_f1(head, gold))
                scores["algebra"].append(token_f1(pred or head, gold))
                scores["same"].append(token_f1(same if how in {"follow", "bridge"} else head, gold))
            print(
                f"  {setting:12s} head {np.mean(scores['head']):.3f}"
                f"  algebra {np.mean(scores['algebra']):.3f}"
                f"  same-pages {np.mean(scores['same']):.3f}"
                f"  {fired}",
                flush=True,
            )
            for left, right, label in (
                ("algebra", "head", "algebra-head"),
                ("same", "head", "same-head"),
                ("algebra", "same", "algebra-same"),
            ):
                delta, lo, hi = bootstrap(scores[left], scores[right])
                mark = " *" if lo > 0 else ""
                print(f"    {label} {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}", flush=True)


if __name__ == "__main__":
    main()
