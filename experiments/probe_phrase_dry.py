"""Print governing phrases. No torch."""
import random
import re
import string
import unicodedata

from datasets import load_dataset

from phrase import candidates, inner_question


def normalize_answer(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_example(ex):
    docs = []
    support = set(ex["supporting_facts"]["title"])
    for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
        clean = [re.sub(r"\s+", " ", str(sent)).strip() for sent in sentences]
        docs.append({"title": title, "sentences": [t for t in clean if t], "support": title in support})
    return {"question": ex["question"], "answer": ex["answer"], "type": ex.get("type") or "unknown", "docs": docs}


def supported(item):
    blob = normalize_answer(" ".join(s for d in item["docs"] for s in d["sentences"]))
    ans = normalize_answer(item["answer"])
    return bool(ans) and ans in blob and len(ans.split()) <= 12


def load(dataset):
    raw = (
        load_dataset("hotpot_qa", "distractor", split="validation")
        if dataset == "hotpot"
        else load_dataset("framolfese/2WikiMultihopQA", split="validation")
    )
    parsed = []
    for ex in raw:
        item = parse_example(ex)
        if dataset == "hotpot" and item["type"] != "bridge":
            continue
        if supported(item):
            parsed.append(item)
    rng = random.Random(3152)
    rng.shuffle(parsed)
    return parsed[:36], parsed[36:156]


for dataset in ("2wiki", "hotpot"):
    dev, test = load(dataset)
    print("\n====", dataset, "DEV")
    for item in dev:
        if re.search(r"\bor\b", item["question"], flags=re.I):
            print("ORDER", item["question"][:130])
            continue
        rows = candidates(item["question"], item["docs"])
        if not rows:
            print("NONE", item["question"][:130])
            continue
        row = rows[0]
        inner = inner_question(row["page"], row["phrase"], row["head"])
        print(f"H={row['head']!r:22} {inner}")
        print("   ", item["question"][:150])
    headed = direct = none = 0
    for item in test:
        rows = candidates(item["question"], item["docs"])
        if not rows:
            none += 1
        elif rows[0]["head"]:
            headed += 1
        else:
            direct += 1
    print(dataset, "test headed", headed, "direct", direct, "none", none)
