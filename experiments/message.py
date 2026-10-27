"""A specialist reads only the pages it holds.

The composer sees titles as addresses and spans as messages. It never receives
a sentence. The message is the span that binds the phrase the question already
wrote; the specialist who holds the page that span names is then asked the
question with that phrase replaced.
"""

import random
import re

from phrase import candidates, governing, inner_question, residual_question

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
EARLIER = re.compile(r"\b(earlier|older|first|before)\b", re.I)
LATER = re.compile(r"\b(later|younger|after)\b", re.I)


def _tools():
    from denote import pointed_year
    from diag_compare2 import choices_of, find_doc
    from run_trustmem import normalize_answer

    return choices_of, find_doc, pointed_year, normalize_answer


def partition(docs, rng):
    ordered = list(docs)
    rng.shuffle(ordered)
    groups = [[], []]
    for index, doc in enumerate(ordered):
        groups[index % 2].append(doc)
    return groups


class Desk:
    """Two specialists. ask() is the only way to read a page."""

    def __init__(self, docs, read, qid):
        self.docs = docs
        self.read = read
        self.groups = partition(docs, random.Random(f"{qid}:2:random"))
        self.owner = {}
        for index, group in enumerate(self.groups):
            for doc in group:
                self.owner[doc["title"]] = (index, doc)

    def ask(self, title, question):
        pair = self.owner.get(title)
        if pair is None or not question:
            return "", 0.0
        _index, doc = pair
        return self.read(question, doc)

    def ask_year(self, title, question):
        _choices, _find, pointed_year, _norm = _tools()
        pair = self.owner.get(title)
        if pair is None:
            return None
        _index, doc = pair
        return pointed_year(self.read, question, doc)


def _same(pred, title, normalize_answer):
    if not pred or not title:
        return False
    return normalize_answer(pred) == normalize_answer(title.split("(")[0])


def _page_name(doc):
    return re.sub(r"\s*\([^)]*\)", "", doc["title"]).strip()


def compare(question, docs, desk):
    choices_of, find_doc, _pointed, normalize_answer = _tools()
    choices = choices_of(question)
    if len(choices) != 2:
        return None
    pages = [find_doc(docs, choice) for choice in choices]
    if any(page is None for page in pages):
        return None
    operands = []
    for choice, page in zip(choices, pages):
        where = page["title"]
        phrase, head, _surface = governing(question, page["title"])
        if head:
            span, _conf = desk.ask(page["title"], inner_question(_page_name(page), phrase, head))
            target = find_doc(docs, span) if span else None
            if target is not None and target["title"] != page["title"]:
                where = target["title"]
        year = desk.ask_year(where, question)
        if year is None:
            return None
        operands.append({"choice": choice, "year": year, "page": where})
    direction = question
    for choice in choices:
        direction = re.sub(re.escape(choice), " ", direction, flags=re.I)
    if EARLIER.search(direction) and not LATER.search(direction):
        winner = min(operands, key=lambda item: item["year"])
    elif LATER.search(direction) and not EARLIER.search(direction):
        winner = max(operands, key=lambda item: item["year"])
    else:
        return None
    asks_year = bool(re.search(r"\byear\b", direction, flags=re.I))
    value = str(winner["year"]) if asks_year else winner["choice"]
    return value, "order", [item["page"] for item in operands], 1.0


def reduce_question(question, docs, desk):
    choices_of, find_doc, _pointed, normalize_answer = _tools()
    if len(choices_of(question)) == 2:
        return "", "", [], 0.0
    rows = candidates(question, docs)
    if not rows:
        return "", "", [], 0.0
    row = rows[0]
    anchor = row["doc"]["title"]
    if not row["head"]:
        span, conf = desk.ask(anchor, question)
        if not span or _same(span, anchor, normalize_answer):
            return "", "", [], 0.0
        return span, "denote", [anchor], conf
    span, conf = desk.ask(anchor, inner_question(row["page"], row["phrase"], row["head"]))
    if not span or _same(span, anchor, normalize_answer):
        return "", "", [], 0.0
    target = find_doc(docs, span)
    pages = [anchor] + ([target["title"]] if target is not None else [])
    identity = question.lower().startswith("who") and normalize_answer(row["phrase"]) in normalize_answer(question)
    if target is not None:
        step, step_conf = desk.ask(
            target["title"],
            f"Who is {_page_name(target)}'s {row['head']}?",
        )
        landed = find_doc(docs, step) if step else None
        if (
            step
            and step_conf > conf
            and not _same(step, target["title"], normalize_answer)
            and normalize_answer(step) != normalize_answer(span)
            and (landed is None or identity)
        ):
            span, conf = step, step_conf
            if landed is not None and landed["title"] not in pages:
                pages.append(landed["title"])
            target = landed or target
    if identity or target is None:
        if question.lower().startswith("who") and span:
            return span, "denote", pages, conf
        if target is None:
            return "", "", [], 0.0
    residual = residual_question(question, row["phrase"], span)
    if normalize_answer(residual) == normalize_answer(span):
        return span, "denote", pages, conf
    pred, pred_conf = desk.ask(target["title"], residual)
    if pred and not _same(pred, target["title"], normalize_answer) and normalize_answer(pred) != normalize_answer(span):
        return pred, "denote", pages, pred_conf
    if question.lower().startswith("who"):
        return span, "denote", pages, conf
    return "", "", [], 0.0


def deliver(question, docs, read, qid):
    """Return value, kind, pages, confidence. The read callback is used only inside Desk.ask."""
    desk = Desk(docs, read, qid)
    choices_of, _find, _pointed, _norm = _tools()
    if len(choices_of(question)) == 2:
        ordered = compare(question, docs, desk)
        if ordered:
            return ordered
        return "", "", [], 0.0
    return reduce_question(question, docs, desk)


def carry(question, docs, read, qid):
    """Follow a span, but ask the original question on the page it names.

    The first message is the best answer either specialist can give on the
    pages it holds. When that span names the other specialist's page, that
    specialist answers the original question there, and the answer replaces
    the first message. The phrase is not rewritten, so a second look at a
    page already scored cannot win by confidence alone.
    """
    _choices, find_doc, _pointed, normalize_answer = _tools()
    desk = Desk(docs, read, qid)
    best = (0.0, "", "")
    for group in desk.groups:
        for doc in group:
            span, conf = desk.ask(doc["title"], question)
            if conf > best[0]:
                best = (conf, span, doc["title"])
    target = find_doc(docs, best[1]) if best[1] else None
    if target is None or target["title"] == best[2]:
        return best[1], best[0]
    span, conf = desk.ask(target["title"], question)
    if span and not _same(span, target["title"], normalize_answer):
        return span, conf
    return best[1], best[0]
