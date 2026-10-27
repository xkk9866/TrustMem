"""Group questions by the operation a witness algebra would run."""

import re
from collections import Counter

from run_durable import load_slices


def kind(question):
    q = question.lower().strip()
    if re.search(r",\s*.+\s+or\s+.+\??$", question.strip(), flags=re.I):
        return "compare"
    if "director" in q and any(w in q for w in ("born", "die", "died", "death")):
        return "director-hop"
    if "director" in q:
        return "director-other"
    if any(w in q for w in ("father", "mother", "spouse", "husband", "wife", "uncle", "aunt")):
        return "kin"
    if q.startswith("where "):
        return "where"
    if q.startswith("when ") or "what year" in q or "which year" in q:
        return "when"
    if q.startswith("who "):
        return "who"
    if q.startswith("which "):
        return "which"
    return "other"


def main():
    for dataset in ("2wiki", "hotpot"):
        _, test, _ = load_slices(dataset, 120, 4)
        counts = Counter(kind(item["question"]) for item in test)
        print("\n==", dataset, counts)
        shown = Counter()
        for item in test:
            key = kind(item["question"])
            if shown[key] >= 3:
                continue
            shown[key] += 1
            print(f"  [{key}] {item['answer']} || {item['question']}")


if __name__ == "__main__":
    main()
