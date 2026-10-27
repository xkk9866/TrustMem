"""The phrase a question already uses to name one page.

The head is a word of that phrase. Work-type words such as film or book are
not the head; they only name the kind of page. Nothing here is a list of
family or professional relations.
"""

import re
import unicodedata

TYPE_WORDS = {
    "film", "films", "song", "songs", "movie", "movies", "book", "books",
    "novel", "novels", "story", "stories", "series", "short",
}
FUNCTION = {"the", "a", "an", "of", "and", "who", "is", "was", "were", "that", "which"}
NONHEAD = FUNCTION | {
    "after", "before", "during", "where", "when", "while", "by", "in", "on",
    "for", "from", "to", "with", "into", "over", "under", "or", "later",
    "earlier", "older", "younger", "first", "which", "whose", "whom", "what",
    "how", "did", "does", "do",
}


def _fold_map(text):
    chars, mapping = [], []
    for index, char in enumerate(text):
        base = unicodedata.normalize("NFKD", char)
        base = "".join(piece for piece in base if not unicodedata.combining(piece))
        if not base:
            continue
        for piece in base:
            chars.append(piece)
            mapping.append(index)
    return "".join(chars), mapping


def title_span(question, title):
    """Return [start, end) of the parenthetical-free title inside the question."""
    stem = re.sub(r"\s*\([^)]*\)", " ", title)
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = re.sub(r"^(the|a|an)\s+", "", stem, flags=re.I).strip()
    letters = re.findall(r"[^\W\d_]", stem, flags=re.UNICODE)
    if len(letters) <= 3:
        return None
    folded_q, mapping = _fold_map(question)
    folded_stem, _ = _fold_map(stem)
    parts = [re.escape(word) for word in folded_stem.split()]
    if not parts:
        return None
    match = re.search(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", folded_q, flags=re.I)
    if not match:
        return None
    start = mapping[match.start()]
    end = mapping[match.end() - 1] + 1
    return start, end


def _content(words):
    return [word for word in words if word.lower() not in NONHEAD and word.lower() not in TYPE_WORDS]


def governing(question, title):
    """Phrase around the title, and the head word the question itself uses."""
    span = title_span(question, title)
    if span is None:
        return "", "", ""
    start, end = span
    surface = re.sub(r"\s+", " ", question[start:end]).strip()
    before = question[:start]
    dets = list(re.finditer(r"\b(the|a|an)\b", before, flags=re.I))
    phrase_start = dets[-1].start() if dets else start
    for _ in range(3):
        interior = question[phrase_start:start]
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", interior)
        if _content(words) or not dets:
            break
        earlier = [item for item in dets if item.start() < phrase_start]
        if not earlier:
            break
        phrase_start = earlier[-1].start()
    possessive = re.match(
        r"'s\s+[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3}",
        question[end:],
        flags=re.I,
    )
    if possessive:
        possessed = possessive.group(0)
        if re.match(r"\s*(when|where|what|how)\b", question, flags=re.I):
            body = possessed[2:].strip()
            pieces = body.split()
            if len(pieces) >= 2 and pieces[-1].lower() == question.strip(" ?").split()[-1].lower():
                possessed = "'s " + " ".join(pieces[:-1])
        phrase_start = start
        phrase_end = end + len(possessed)
    else:
        phrase_end = end
    phrase = re.sub(r"\s+", " ", question[phrase_start:phrase_end]).strip(" ,")
    leftover = re.sub(re.escape(surface), " ", phrase, flags=re.I)
    leftover = re.sub(r"'s\b", " ", leftover, flags=re.I)
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", leftover)
    content = _content(words)
    if not content:
        return phrase, "", surface
    # The head is the governing noun plus the modifiers written immediately before it.
    head_at = max(index for index, word in enumerate(words) if word == content[-1])
    chunk = [content[-1]]
    for word in reversed(words[:head_at]):
        if word.lower() in NONHEAD or word.lower() in TYPE_WORDS:
            break
        chunk.append(word)
    head = " ".join(reversed(chunk))
    attached = bool(re.search(r"\bof\b", phrase, flags=re.I)) or bool(
        re.search(re.escape(surface) + r"\s*'s\b", phrase, flags=re.I)
    )
    if not attached:
        head = ""
    return phrase, head, surface


def inner_question(page, phrase, head):
    if not head:
        return ""
    if re.search(r"'s\s+" + re.escape(head) + r"\s*$", phrase, flags=re.I):
        return f"Who is {page}'s {head}?"
    return f"Who is the {head} of {page}?"


def residual_question(question, phrase, witness):
    text = re.sub(re.escape(phrase), witness, question, count=1, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("?"):
        text += "?"
    return text


def _mentions_title(head, docs, own_title):
    folded = " ".join(head.lower().split())
    if len(folded) <= 3:
        return False
    for doc in docs:
        if doc["title"] == own_title:
            continue
        stem = re.sub(r"\s*\([^)]*\)", " ", doc["title"]).lower()
        if folded == " ".join(stem.split()) or folded in stem:
            return True
    return False


def candidates(question, docs):
    """Titled pages named by the question, shortest real phrase first."""
    rows = []
    for doc in docs:
        phrase, head, surface = governing(question, doc["title"])
        if not surface:
            continue
        if head and _mentions_title(head, docs, doc["title"]):
            head = ""
        rows.append({
            "doc": doc,
            "phrase": phrase,
            "head": head,
            "surface": surface,
            "page": re.sub(r"\s*\([^)]*\)", "", doc["title"]).strip(),
        })
    def mentioned(title):
        bits = re.findall(r"\(([^)]*)\)", title)
        return any(bit.lower() in question.lower() for bit in bits)

    headed = [row for row in rows if row["head"]]
    pool = headed or rows
    pool.sort(key=lambda row: (0 if mentioned(row["doc"]["title"]) else 1, -len(row["surface"]), len(row["phrase"])))
    return pool
