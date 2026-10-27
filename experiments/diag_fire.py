"""Why comparison returns an answer on only 14 of the choice questions."""

import pickle
import re
from collections import Counter

from run_durable import load_slices
from run_trustmem import normalize_answer

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
RANGE = re.compile(r"(1[0-9]{3})\s*[–—-]\s*(1[0-9]{3}|20[0-9]{2})")


def choices_of(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [match.group(1).strip(), match.group(2).strip()]


def find_doc(docs, name):
    needle = normalize_answer(name)
    if not needle:
        return None
    for doc in docs:
        title = normalize_answer(doc["title"])
        if title == needle or title.startswith(needle) or needle.startswith(title):
            return doc
    return None


def main():
    _, test, _ = load_slices("2wiki", 120, 4)
    by_q = {item["question"]: item for item in test}
    reasons = Counter()
    shown = 0
    for item in test:
        opts = choices_of(item["question"])
        if len(opts) != 2:
            reasons["no-choice"] += 1
            continue
        docs = item["docs"]
        pages = [find_doc(docs, opt) for opt in opts]
        if any(page is None for page in pages):
            reasons["missing-page"] += 1
            if shown < 4:
                shown += 1
                print("missing", opts, [doc["title"] for doc in docs[:6]])
            continue
        q = item["question"].lower()
        if "director" in q:
            for page in pages:
                lede = page["sentences"][0] if page["sentences"] else ""
                if "directed by" not in lede.lower() and shown < 8:
                    shown += 1
                    print("lede", page["title"], "||", lede[:180])
            reasons["director-q"] += 1
        elif any(word in q for word in ("died", "born", "longer")):
            lede = pages[0]["sentences"][0][:180] if pages[0]["sentences"] else ""
            if shown < 8:
                shown += 1
                print("person", item["question"], "||", lede)
            reasons["person-q"] += 1
        else:
            reasons["release-q"] += 1
    print(reasons)


if __name__ == "__main__":
    main()
