"""Can a comparison of values on the two cited pages answer an explicit choice?"""

import pickle
import re
from pathlib import Path

from run_durable import load_slices
from run_trustmem import normalize_answer

ROOT = Path(__file__).resolve().parents[1]
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def choices(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [match.group(1).strip(), match.group(2).strip()]


def direction(question):
    q = question.lower()
    if any(w in q for w in ("more recently", "later", "after", "younger")):
        return "max"
    if any(w in q for w in ("earlier", "first", "older", "before", "longer")):
        return "min"
    return ""


def years_for(choice, docs):
    needle = normalize_answer(choice)
    found = []
    for doc in docs:
        blob = normalize_answer(" ".join(doc["sentences"]))
        title = normalize_answer(doc["title"])
        if needle and (needle in title or needle in blob):
            for sent in doc["sentences"]:
                if needle in normalize_answer(sent) or needle in title:
                    found.extend(int(y) for y in YEAR.findall(sent))
    return found


def main():
    _, test, _ = load_slices("2wiki", 120, 4)
    by_q = {item["question"]: item for item in test}
    with (ROOT / "results" / "packets_2wiki.pkl").open("rb") as handle:
        _, _, meta = pickle.load(handle)
    hit = miss = skip = 0
    for item in meta:
        opts = choices(item["question"])
        way = direction(item["question"])
        if len(opts) != 2 or not way:
            continue
        docs = by_q[item["question"]]["docs"]
        left, right = years_for(opts[0], docs), years_for(opts[1], docs)
        if not left or not right:
            skip += 1
            if skip <= 6:
                print("skip", item["answer"], "||", item["question"])
                print("   years", left, right)
            continue
        if "longer" in item["question"].lower() or "older" in item["question"].lower():
            pick = opts[0] if (max(left) - min(left)) >= (max(right) - min(right)) else opts[1]
            if way == "min" and "older" in item["question"].lower():
                pick = opts[0] if min(left) <= min(right) else opts[1]
        elif way == "min":
            pick = opts[0] if min(left) <= min(right) else opts[1]
        else:
            pick = opts[0] if max(left) >= max(right) else opts[1]
        ok = normalize_answer(pick) == normalize_answer(item["answer"])
        hit += int(ok)
        miss += int(not ok)
        if not ok and miss <= 8:
            print("miss", item["answer"], "picked", pick, "||", item["question"])
            print("   years", left[:6], right[:6])
    print(f"correct {hit} wrong {miss} no-year {skip}")


if __name__ == "__main__":
    main()
