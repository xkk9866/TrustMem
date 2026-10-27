"""How often a title or a mentioned page reaches the answer, without a reader."""

import re
from collections import Counter

from diag_compare2 import pick
from run_durable import load_slices
from run_trustmem import normalize_answer, tokens

PERSON = r"([A-Z][A-Za-zÁÉÍÓÚÜÑáéíóúüñ.'’\-]+(?:\s+[A-Z][A-Za-zÁÉÍÓÚÜÑáéíóúüñ.'’\-]+){0,4})"


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


def mentioned(page, docs):
    if page is None:
        return []
    blob = " ".join(page["sentences"][:3]).lower()
    hits = []
    for doc in docs:
        if doc["title"] == page["title"]:
            continue
        title = doc["title"].strip()
        if len(title) > 3 and title.lower() in blob:
            hits.append(doc)
    hits.sort(key=lambda doc: len(doc["title"]), reverse=True)
    return hits


def best_page(question, docs):
    q = tokens(question)
    best, score = None, 0
    for doc in docs:
        lede = doc["sentences"][0] if doc["sentences"] else ""
        overlap = len(q & tokens(f"{doc['title']} {lede}"))
        if overlap > score:
            best, score = doc, overlap
    return best, score


def answer_on(doc, answer):
    if doc is None:
        return False
    blob = normalize_answer(" ".join(doc["sentences"]))
    return normalize_answer(answer) in blob


def main():
    for dataset in ("hotpot", "2wiki"):
        _, test, _ = load_slices(dataset, 120, 4)
        counts = Counter()
        shown = 0
        for item in test:
            question, answer, docs = item["question"], item["answer"], item["docs"]
            counts["n"] += 1
            titled = anchor(question, docs)
            counts["title_anchor"] += titled is not None
            if pick(question, docs):
                counts["compare_fires"] += 1
            person = bridge(question, titled) if titled is not None else ""
            counts["relation_string"] += bool(person)
            page, score = best_page(question, docs)
            counts["overlap_ge2"] += score >= 2
            links = mentioned(page, docs)
            counts["has_mention"] += bool(links)
            support = [doc for doc in docs if doc["support"]]
            counts["answer_on_overlap"] += answer_on(page, answer)
            counts["answer_on_mention"] += any(answer_on(doc, answer) for doc in links)
            counts["answer_on_title_anchor"] += answer_on(titled, answer)
            linked_from_support = []
            for src in support:
                linked_from_support.extend(mentioned(src, docs))
            counts["gold_link_has_answer"] += any(answer_on(doc, answer) for doc in linked_from_support)
            if shown < 6 and dataset == "hotpot" and titled is None:
                shown += 1
                print(
                    f"  overlap={score} mentions={len(links)} ans_on_overlap={answer_on(page, answer)} "
                    f"ans_on_mention={any(answer_on(doc, answer) for doc in links)}"
                )
                print("   Q", question)
                print("   A", answer, "| page", None if page is None else page["title"])
                print("   links", [doc["title"] for doc in links[:4]])
        print(dataset, dict(counts))


if __name__ == "__main__":
    main()
