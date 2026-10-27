"""Inside-out reduction of a question across specialists.

The question names a page. The shortest noun phrase that contains that page
is what the holder of the page can bind. The holder sends the binding, which
is a span, not its sentences. The specialist who holds the page named by the
binding receives the question with that phrase replaced by the span, and
reads only that page. The head noun is whatever word the question itself
uses. There is no list of relations.
"""

import re

from diag_compare2 import choices_of, find_doc
from run_trustmem import normalize_answer

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
EARLIER = re.compile(r"\b(earlier|older|first|before)\b", re.I)
LATER = re.compile(r"\b(later|younger|after)\b", re.I)


def _same(pred, title):
    return bool(pred) and normalize_answer(pred) == normalize_answer(title.split("(")[0])


def named_pages(question, docs):
    hits = [doc for doc in docs if len(doc["title"]) > 3 and doc["title"].lower() in question.lower()]
    hits.sort(key=lambda doc: len(doc["title"]), reverse=True)
    return hits


def containing_phrase(question, title):
    """The shortest determiner phrase in the question that contains the title."""
    idx = question.lower().find(title.lower())
    if idx < 0:
        return title
    before = question[:idx]
    starts = list(re.finditer(r"\b(the|a|an)\b", before, re.I))
    start = starts[-1].start() if starts else idx
    end = idx + len(title)
    possessive = re.match(
        r"'s\s+[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3}",
        question[end:],
        flags=re.I,
    )
    if possessive and start == idx:
        end = end + possessive.end()
    return re.sub(r"\s+", " ", question[start:end]).strip(" ,")


def head_word(phrase, title):
    rest = re.sub(re.escape(title), " ", phrase, flags=re.I)
    rest = re.sub(r"\b(the|a|an|of|film|song|movie|who|is|was|were)\b", " ", rest, flags=re.I)
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", rest)
    return words[-1] if words else ""


def inner_question(phrase, title, head):
    if head:
        return f"Who is the {head} of {title}?"
    cleaned = phrase if phrase.endswith("?") else f"Who is {phrase}?"
    return cleaned


def residual_question(question, phrase, witness):
    text = re.sub(re.escape(phrase), witness, question, count=1, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("?"):
        text += "?"
    return text


def _year(span):
    match = YEAR.search(span or "")
    return int(match.group(1)) if match else None


def _asks_year(question, choices):
    text = question
    for choice in choices:
        text = re.sub(re.escape(choice), " ", text, flags=re.I)
    return bool(re.search(r"\byear\b", text, re.I))


def pointed_year(read, question, doc):
    """The year the reader points at on this page. No event lexicon."""
    scorer = getattr(read, "score_span", None)
    if scorer is None or not doc:
        return None
    best = (0.0, None)
    for sentence in doc["sentences"][:4]:
        context = f"{doc['title']}: {sentence}"
        for year in YEAR.findall(sentence):
            score = scorer(question, context, year)
            if score > best[0]:
                best = (score, int(year))
    return best[1]


def comparison(question, docs, read):
    """A coordinated question has two arguments. Each page witnesses one year."""
    choices = choices_of(question)
    if len(choices) != 2:
        return None
    pages = [find_doc(docs, choice) for choice in choices]
    if any(page is None for page in pages):
        return None
    operands = []
    for choice, page in zip(choices, pages):
        where = page
        phrase = containing_phrase(question, page["title"])
        head = head_word(phrase, page["title"])
        if head:
            bound, _ = read(inner_question(phrase, page["title"], head), page)
            target = find_doc(docs, bound) if bound else None
            if target is not None and target["title"] != page["title"]:
                where = target
        year = pointed_year(read, question, where)
        if year is None:
            return None
        operands.append({"choice": choice, "year": year, "page": where["title"]})
    direction = question
    for choice in choices:
        direction = re.sub(re.escape(choice), " ", direction, flags=re.I)
    if EARLIER.search(direction) and not LATER.search(direction):
        winner = min(operands, key=lambda item: item["year"])
    elif LATER.search(direction) and not EARLIER.search(direction):
        winner = max(operands, key=lambda item: item["year"])
    else:
        return None
    value = str(winner["year"]) if _asks_year(question, choices) else winner["choice"]
    return value, "order", [item["page"] for item in operands], 1.0


def reduce_question(question, docs, read):
    """Bind the innermost titled phrase, then read what remains on that page."""
    named = named_pages(question, docs)
    if not named:
        return "", "", [], 0.0
    anchor = named[0]
    phrase = containing_phrase(question, anchor["title"])
    head = head_word(phrase, anchor["title"])
    if not head:
        return "", "", [], 0.0
    span, conf = read(inner_question(phrase, anchor["title"], head), anchor)
    if not span or _same(span, anchor["title"]):
        return "", "", [], 0.0
    target = find_doc(docs, span)
    pages = [anchor["title"]] + ([target["title"]] if target else [])
    identity = question.lower().startswith("who") and normalize_answer(containing_phrase(question, anchor["title"])) in normalize_answer(question)
    if target is not None and head:
        step, step_conf = read(f"Who is the {head} of {span}?", target)
        landed = find_doc(docs, step) if step else None
        if (
            step
            and step_conf > conf
            and not _same(step, target["title"])
            and normalize_answer(step) != normalize_answer(span)
            and (landed is None or identity)
        ):
            span, conf = step, step_conf
            if landed is not None:
                pages.append(landed["title"])
                target = landed
    if identity or target is None:
        if question.lower().startswith("who") and span:
            return span, "denote", pages, conf
        if target is None:
            return "", "", [], 0.0
    residual = residual_question(question, phrase, span)
    if normalize_answer(residual) == normalize_answer(span):
        return span, "denote", pages, conf
    pred, pred_conf = read(residual, target)
    if pred and not _same(pred, target["title"]) and normalize_answer(pred) != normalize_answer(span):
        return pred, "denote", pages, pred_conf
    if question.lower().startswith("who"):
        return span, "denote", pages, conf
    return "", "", [], 0.0


def denote(question, docs, read):
    """Return value, kind, pages, confidence. kind is order, denote, or empty."""
    if len(choices_of(question)) == 2:
        ordered = comparison(question, docs, read)
        if ordered:
            return ordered
        return "", "", [], 0.0
    return reduce_question(question, docs, read)
