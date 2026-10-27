"""Trace the younger-director comparison."""

from diag_compare2 import birth_only, choices_of, director_doc, emit, find_doc, pick
from run_durable import load_slices


def main():
    _, test, _ = load_slices("2wiki", 120, 4)
    item = next(ex for ex in test if "Deadly Eyes" in ex["question"])
    question, docs = item["question"], item["docs"]
    opts = choices_of(question)
    q = question.lower()
    pages = [find_doc(docs, opt) for opt in opts]
    people = [director_doc(docs, page) for page in pages]
    values = [birth_only(page) for page in people]
    want_min = "later" not in q and "younger" not in q
    print("opts", opts)
    print("films", [None if page is None else page["title"] for page in pages])
    print("people", [None if page is None else page["title"] for page in people])
    print("values", values, "want_min", want_min)
    print("emit", emit(question, opts, values, want_min))
    print("pick", pick(question, docs))


if __name__ == "__main__":
    main()
