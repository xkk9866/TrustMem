"""One evaluation of a question over page-local witnesses.

A question denotes a page. Reading that page either yields comparable values
for two denoted pages, or a span that names a second page. The original
question is then read on that second page. If neither happens, the derivation
does not replace the packet.
"""

import re

from diag_compare2 import find_doc, pick, support_pages
from run_trustmem import normalize_answer

STOP = set(
    "a an the of to in on for and or is was were be been being by with from as at "
    "that this it its their his her she he they them who whom which what when where "
    "how why into over under about than then so if not no yes do does did film song "
    "director performer founder born birth died die death".split()
)


def titles_named(question, docs):
    folded = question.lower()
    hits = [doc for doc in docs if len(doc["title"]) > 3 and doc["title"].lower() in folded]
    hits.sort(key=lambda doc: len(doc["title"]), reverse=True)
    return hits


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def described_page(question, docs):
    """The page a description denotes when the question names no title.

    The winner must strictly outscore every other page. A tie denotes nothing.
    """
    query = _tokens(question)
    if len(query) < 3:
        return None
    scored = []
    for doc in docs:
        body = " ".join(doc["sentences"][:4])
        overlap = len(query & _tokens(body))
        scored.append((overlap, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 4:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None
    return scored[0][1]


def _same(pred, title):
    return bool(pred) and normalize_answer(pred) == normalize_answer(title.split("(")[0])


def outer_question(question, anchor_title, witness):
    """Ask the question that remains after the witness replaces the denoted page.

    The witness stands in for the page that introduced it. A relation that has
    already been read is removed, so the second page is asked for the outer
    predicate only. An in-law is the parent of the spouse the first page named.
    """
    low = question.lower()
    if "father-in-law" in low:
        return f"Who is the father of {witness}?"
    if "mother-in-law" in low:
        return f"Who is the mother of {witness}?"
    if "grandfather" in low:
        return f"Who is the father of {witness}?"
    text = question
    if anchor_title and anchor_title.lower() in low:
        text = re.sub(re.escape(anchor_title), witness, question, count=1, flags=re.I)
    text = re.sub(
        r"\b(?:the|a|an)\s+(?:film\s+|song\s+|movie\s+)?"
        r"(?:director|founder|performer|singer|composer|wife|husband|spouse|"
        r"mother|father|son|daughter|writer|author|producer)\s+of\s+"
        r"(?:the\s+)?(?:film\s+|song\s+|movie\s+)?",
        "",
        text,
        count=1,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if "born" in low and witness.lower() not in text.lower():
        return f"Where was {witness} born?"
    if re.search(r"\b(nationality|country)\b", low) and witness.lower() not in text.lower():
        return f"What country is {witness} from?"
    if not text.endswith("?"):
        text += "?"
    return text


def inner_question(question, anchor_title):
    """The first page is asked who it contributes, not the outer predicate.

    The birthplace is not on the film page. Asking the original question there
    returns a word from the film sentence. The inner question asks for the
    person the question's own relation names.
    """
    low = question.lower()
    if "father-in-law" in low or "mother-in-law" in low:
        return f"Who is the spouse of {anchor_title}?"
    if "grandfather" in low:
        return f"Who is the father of {anchor_title}?"
    relations = (
        "director",
        "composer",
        "performer",
        "singer",
        "founder",
        "wife",
        "husband",
        "spouse",
        "mother",
        "father",
        "producer",
        "writer",
        "author",
    )
    for relation in relations:
        if re.search(rf"\b{relation}\b", low):
            return f"Who is the {relation} of {anchor_title}?"
    return question


def derive(question, docs, read):
    """Return (value, kind, pages, confidence).

    kind is order, pointer, or empty. empty means the packet should stand.
    read(question, doc) -> (span, confidence).
    A pointer is followed only from a title the question itself names.
    """
    ordered = pick(question, docs)
    if ordered:
        return ordered, "order", support_pages(question, docs), 1.0

    named = titles_named(question, docs)
    if not named:
        return "", "", [], 0.0
    anchor = named[0]
    span, _ = read(inner_question(question, anchor["title"]), anchor)
    target = find_doc(docs, span) if span else None
    if target is None or target["title"] == anchor["title"]:
        return "", "", [], 0.0

    asked = outer_question(question, anchor["title"], span)
    pred, conf = read(asked, target)
    if pred and not _same(pred, target["title"]) and normalize_answer(pred) != normalize_answer(span):
        return pred, "pointer", [anchor, target], conf
    if question.lower().startswith("who") and not _same(span, anchor["title"]):
        return span, "pointer", [anchor, target], conf
    return "", "", [], 0.0
