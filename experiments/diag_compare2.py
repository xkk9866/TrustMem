"""Cleaner value extraction for an explicit two-candidate question."""

import pickle
import re

from run_durable import load_slices
from run_trustmem import normalize_answer, token_f1

YEAR = re.compile(r"(1[0-9]{3}|20[0-9]{2})")
SPAN = re.compile(
    r"(1[0-9]{3}|20[0-9]{2}).{0,50}?[–—-].{0,40}?(1[0-9]{3}|20[0-9]{2})"
)
DIRECTOR = re.compile(r"(?:directed by|film by|filmed by)\s+(?:author\s+|filmmaker\s+)?([A-Z][A-Za-z.'’\- ]{1,40})")


def choices_of(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [match.group(1).strip(), match.group(2).strip()]


def clean_name(name):
    name = re.split(r"\s+(?:and|who|with|from|for|starring)\s+", name.strip())[0]
    return " ".join(name.split()[:5])


def find_doc(docs, name):
    needle = normalize_answer(clean_name(name))
    if len(needle) < 4:
        return None
    exact = None
    best, score = None, 0
    for doc in docs:
        title = normalize_answer(doc["title"])
        if len(title) < 4:
            continue
        if title == needle:
            exact = doc
            break
        if title.startswith(needle) or needle.startswith(title):
            overlap = min(len(title), len(needle))
            if overlap > score:
                best, score = doc, overlap
    return exact or best


def lifespan(doc):
    if not doc or not doc["sentences"]:
        return None
    match = SPAN.search(doc["sentences"][0])
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def birth_only(doc):
    if not doc or not doc["sentences"]:
        return None
    match = re.search(r"born.{0,40}?(1[0-9]{3}|20[0-9]{2})", doc["sentences"][0], re.I)
    if match:
        return int(match.group(1))
    pair = lifespan(doc)
    return pair[0] if pair else None


def death_only(doc):
    pair = lifespan(doc)
    if pair:
        return pair[1]
    if not doc or not doc["sentences"]:
        return None
    match = re.search(r"died.{0,40}?(1[0-9]{3}|20[0-9]{2})", doc["sentences"][0], re.I)
    return int(match.group(1)) if match else None


def release_year(doc):
    if not doc:
        return None
    for sent in doc["sentences"][:4]:
        match = re.search(
            r"(?:released|premiered|opened|founded|published).{0,40}?(1[0-9]{3}|20[0-9]{2})",
            sent,
            re.I,
        )
        if match:
            return int(match.group(1))
        match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b.{0,30}(?:film|album|novel|song|school|documentary)", sent, re.I)
        if match:
            return int(match.group(1))
        match = re.search(r"\(([^)]*?(1[0-9]{3}|20[0-9]{2}))", sent)
        if match:
            return int(match.group(2))
        match = re.search(r"\b(?:a|an|the)\s+(1[0-9]{3}|20[0-9]{2})\b", sent, re.I)
        if match:
            return int(match.group(1))
    return None


def director_doc(docs, film):
    if not film:
        return None
    blob = " ".join(film["sentences"][:3])
    match = DIRECTOR.search(blob)
    if not match:
        return None
    return find_doc(docs, clean_name(match.group(1)))


def asks_year(question, opts):
    """The question asks for a year. A year inside a candidate title does not count."""
    q = question.lower()
    if q.startswith("when ") or q.startswith("what year"):
        return True
    stripped = q
    for opt in opts:
        stripped = stripped.replace(opt.lower(), " ")
    return re.search(r"\byear\b", stripped) is not None


def emit(question, opts, values, want_min):
    if any(value is None for value in values):
        return ""
    index = min(range(2), key=lambda i: values[i]) if want_min else max(range(2), key=lambda i: values[i])
    if asks_year(question, opts):
        return str(values[index])
    return opts[index]


def support_pages(question, docs):
    """Pages whose values the comparison actually reads."""
    opts = choices_of(question)
    if len(opts) != 2:
        return []
    pages = [find_doc(docs, opt) for opt in opts]
    if "director" in question.lower():
        pages = [director_doc(docs, page) for page in pages]
    return [page for page in pages if page is not None]


def has_word(text, *words):
    return any(re.search(rf"\b{word}\b", text) for word in words)


def pick(question, docs):
    opts = choices_of(question)
    if len(opts) != 2:
        return ""
    q = question.lower()
    pages = [find_doc(docs, opt) for opt in opts]
    if "director" in q:
        pages = [director_doc(docs, page) for page in pages]
        if any(page is None for page in pages):
            return ""
        if has_word(q, "died", "dead"):
            values = [death_only(page) for page in pages]
            want_min = not has_word(q, "later")
        elif has_word(q, "born", "older", "younger"):
            values = [birth_only(page) for page in pages]
            want_min = not has_word(q, "later", "younger")
        else:
            return ""
    elif "lived longer" in q or "live longer" in q:
        values = []
        for page in pages:
            pair = lifespan(page)
            values.append(None if pair is None else pair[1] - pair[0])
        if any(value is None for value in values):
            return ""
        return opts[0] if values[0] >= values[1] else opts[1]
    elif has_word(q, "died", "dead"):
        values = [death_only(page) for page in pages]
        want_min = not has_word(q, "later")
    elif has_word(q, "born"):
        values = [birth_only(page) for page in pages]
        want_min = not has_word(q, "later", "younger")
    else:
        values = [release_year(page) for page in pages]
        want_min = not has_word(q, "later", "after") and "more recently" not in q
    return emit(question, opts, values, want_min)


def main():
    _, test, _ = load_slices("2wiki", 120, 4)
    fired = correct = wrong = choices = 0
    f1s = []
    for item in test:
        if len(choices_of(item["question"])) == 2:
            choices += 1
        answer = pick(item["question"], item["docs"])
        if not answer:
            continue
        fired += 1
        f1 = token_f1(answer, item["answer"])
        f1s.append(f1)
        correct += int(f1 == 1.0)
        wrong += int(f1 < 1.0)
        if f1 < 1.0 and wrong <= 10:
            print(f"wrong gold={item['answer']!r} pick={answer!r}")
            print("  ", item["question"])
    mean = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"choices {choices} fired {fired} exact {correct} partial-or-wrong {wrong} mean_f1 {mean:.3f}")


if __name__ == "__main__":
    main()
