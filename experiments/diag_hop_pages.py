"""See whether a kin or director fact sits on the named page."""

import re

from diag_compare2 import find_doc
from run_durable import load_slices


def anchor(question, docs):
    qn = question
    hits = []
    for doc in docs:
        title = doc["title"]
        if len(title) > 3 and title.lower() in qn.lower():
            hits.append(doc)
    hits.sort(key=lambda doc: len(doc["title"]), reverse=True)
    return hits[:2]


def main():
    _, test, _ = load_slices("2wiki", 120, 4)
    shown = 0
    for item in test:
        q = item["question"].lower()
        if " or " in q:
            continue
        if not any(w in q for w in ("father", "mother", "director", "composer", "performer", "founder")):
            continue
        if shown >= 8:
            break
        shown += 1
        print("\nQ:", item["question"])
        print("A:", item["answer"])
        for doc in anchor(item["question"], item["docs"]):
            print(" PAGE", doc["title"])
            for sent in doc["sentences"][:2]:
                print("  ", sent[:220])


if __name__ == "__main__":
    main()
