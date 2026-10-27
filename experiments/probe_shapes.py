"""Question shapes. Does not import the reader or torch."""
import random
import re
import string

from datasets import load_dataset

CLAUSE = re.compile(r"\b(who|whom|whose|which|that)\b", re.I)


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_example(ex):
    titles = list(ex["context"]["title"])
    sent_groups = list(ex["context"]["sentences"])
    support_titles = set(ex["supporting_facts"]["title"])
    docs = []
    for i, (title, sentences) in enumerate(zip(titles, sent_groups)):
        clean = [re.sub(r"\s+", " ", str(sent)).strip() for sent in sentences]
        clean = [text for text in clean if text]
        docs.append({"title": title, "sentences": clean, "support": title in support_titles})
    return {
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type") or "unknown",
        "docs": docs,
    }


def answer_supported(item):
    blob = normalize_answer(" ".join(sent for doc in item["docs"] for sent in doc["sentences"]))
    ans = normalize_answer(item["answer"])
    return bool(ans) and ans in blob and len(ans.split()) <= 12


def choices_of(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    return [match.group(1).strip(), match.group(2).strip()] if match else []


def named_pages(question, docs):
    hits = [doc["title"] for doc in docs if len(doc["title"]) > 3 and doc["title"].lower() in question.lower()]
    hits.sort(key=len, reverse=True)
    return hits


def load(dataset, n_test=120):
    if dataset == "hotpot":
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
    else:
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    parsed = []
    for ex in raw:
        item = parse_example(ex)
        if dataset == "hotpot" and item["type"] != "bridge":
            continue
        if answer_supported(item):
            parsed.append(item)
    rng = random.Random(3152)
    rng.shuffle(parsed)
    return parsed[:36], parsed[36:36 + n_test]


for dataset in ("hotpot", "2wiki"):
    dev, test = load(dataset)
    print("\n====", dataset)
    for split, rows in (("DEV", dev), ("TEST", test)):
        counts = {"ORDER": 0, "NAMED": 0, "CLAUSE": 0, "BARE": 0}
        shown = 0
        for item in rows:
            question = item["question"]
            named = named_pages(question, item["docs"])
            if len(choices_of(question)) == 2:
                kind = "ORDER"
            elif named:
                kind = "NAMED"
            elif CLAUSE.search(question):
                kind = "CLAUSE"
            else:
                kind = "BARE"
            counts[kind] += 1
            if split == "DEV" and dataset == "hotpot" and kind != "ORDER" and shown < 28:
                support = [doc["title"] for doc in item["docs"] if doc["support"]]
                print(f"[{kind}] {question}")
                print(f"    gold={item['answer']!r} support={support} named={named}")
                shown += 1
        print(split, counts)
