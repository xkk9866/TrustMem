import pickle
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def choices(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    if not match:
        match = re.search(r"\b(before|after)\b.+\b(before|after)\b", text, flags=re.I)
        if match:
            return ["before", "after"]
        return []
    left, right = match.group(1), match.group(2)
    return [left.strip(), right.strip()]


for name in ("hotpot", "2wiki"):
    with (root / "results" / f"packets_{name}.pkl").open("rb") as handle:
        _, _, meta = pickle.load(handle)
    hit = 0
    gold_in = 0
    for item in meta:
        opts = choices(item["question"])
        if not opts:
            continue
        hit += 1
        ans = item["answer"].strip().lower()
        if any(ans == opt.lower() or ans in opt.lower() or opt.lower() in ans for opt in opts):
            gold_in += 1
    print(f"{name}: choice questions {hit}/{len(meta)}, gold is a choice {gold_in}")
