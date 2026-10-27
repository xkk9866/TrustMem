"""Page-local witnesses.

Comparison orders two values that sit on different pages.
Follow reads the object of a relation, then the asked attribute on that object's page.
Citation opens the pages named inside the page the question itself names.
The packet commitment is the fallback when none of these resolves.
"""

import re

from diag_compare2 import death_only, find_doc, pick, support_pages

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


def cited_pages(question, docs, limit=3):
    """Pages cited by the page whose title the question names.

    The anchor itself is included. Returns nothing when the anchor cites no
    other page in the closed corpus, so the caller keeps the packet.
    """
    page = anchor(question, docs)
    if page is None:
        return []
    blob = " ".join(page["sentences"][:4]).lower()
    links = []
    for doc in docs:
        title = doc["title"].strip()
        if doc["title"] == page["title"] or len(title) <= 3:
            continue
        if title.lower() in blob:
            links.append(doc)
    if not links:
        return []
    links.sort(key=lambda doc: len(doc["title"]), reverse=True)
    return [page] + links[:limit]


def express(question, docs, read):
    """Return (value, how, pages). how is compare, follow, bridge, or empty."""
    judged = pick(question, docs)
    if judged:
        return judged, "compare", support_pages(question, docs)
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
    if not pred or _same(pred, target["title"]):
        return "", "", []
    return pred, "follow", visited


def _same(pred, title):
    from run_trustmem import normalize_answer

    return normalize_answer(pred) == normalize_answer(title)
