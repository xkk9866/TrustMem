"""Dry parse of phrase heads. No torch."""
import random
import re
import string
import unicodedata

from datasets import load_dataset

FUNCTION = {
    "the", "a", "an", "of", "film", "song", "movie", "who", "is", "was", "were",
    "and", "in", "on", "for", "from", "by", "to", "with", "at", "after", "before",
}


def normalize_answer(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def stem(title):
    return re.sub(r"\s*\([^)]*\)\s*", " ", title).strip()


def parse_example(ex):
    titles = list(ex["context"]["title"])
    sent_groups = list(ex["context"]["sentences"])
    docs = []
    for title, sentences in zip(titles, sent_groups):
        clean = [re.sub(r"\s+", " ", str(sent)).strip() for sent in sentences]
        docs.append({"title": title, "sentences": [t for t in clean if t]})
    return {"question": ex["question"], "answer": ex["answer"], "type": ex.get("type") or "unknown", "docs": docs}


def supported(item):
    blob = normalize_answer(" ".join(s for d in item["docs"] for s in d["sentences"]))
    ans = normalize_answer(item["answer"])
    return bool(ans) and ans in blob and len(ans.split()) <= 12


def choices_of(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    return [match.group(1).strip(), match.group(2).strip()] if match else []


def named_pages(question, docs):
    q = normalize_answer(question)
    hits = []
    for doc in docs:
        stem_title = stem(doc["title"])
        key = normalize_answer(stem_title)
        if len(key) > 3 and key in q:
            hits.append(doc)
    hits.sort(key=lambda doc: len(normalize_answer(stem(doc["title"]))), reverse=True)
    return hits


def phrase_and_head(question, title):
    stem_title = stem(title)
    idx = question.lower().find(stem_title.lower())
    if idx < 0:
        # accent-folded search
        folded = normalize_answer(question)
        key = normalize_answer(stem_title)
        pos = folded.find(key)
        if pos < 0:
            return stem_title, ""
        idx = pos
        # approximate: use stem length in original if equal length, else the stem itself
        end = idx + len(stem_title) if question.lower().find(stem_title.lower()) == idx else idx + len(key)
    else:
        end = idx + len(stem_title)
    before = question[:idx]
    starts = list(re.finditer(r"\b(the|a|an)\b", before, re.I))
    # outermost determiner that does not cross a sentence comma-clause
    start = starts[0].start() if starts else idx
    if starts:
        # do not start before the previous comma if that comma is a clause break
        comma = before.rfind(",")
        if comma >= 0:
            later = [m.start() for m in starts if m.start() > comma]
            start = later[0] if later else starts[-1].start()
        else:
            start = starts[0].start()
    phrase = re.sub(r"\s+", " ", question[start:end]).strip(" ,")
    rest = phrase
    cut = re.search(r"\bof\b", rest, re.I)
    if cut:
        left = rest[:cut.start()]
    else:
        left = re.sub(re.escape(stem_title), " ", rest, flags=re.I)
    left = re.sub(r"[^A-Za-z\s'-]", " ", left)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", left) if w.lower() not in FUNCTION]
    head = words[-1] if words else ""
    return phrase, head


def load(dataset):
    raw = load_dataset("hotpot_qa", "distractor", split="validation") if dataset == "hotpot" else load_dataset("framolfese/2WikiMultihopQA", split="validation")
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


for dataset in ("hotpot", "2wiki"):
    dev, test = load(dataset)
    print("\n====", dataset, "dev")
    head_n = direct_n = order_n = none_n = 0
    for item in dev:
        if len(choices_of(item["question"])) == 2:
            order_n += 1
            continue
        named = named_pages(item["question"], item["docs"])
        if not named:
            none_n += 1
            print("NONE", item["question"][:140])
            continue
        phrase, head = phrase_and_head(item["question"], named[0]["title"])
        if head:
            head_n += 1
            print(f"HEAD {head!r} | {phrase[:80]!r}")
            print("   ", item["question"][:140])
        else:
            direct_n += 1
            print("DIRECT", named[0]["title"], "|", item["question"][:120])
    print("order", order_n, "head", head_n, "direct", direct_n, "none", none_n)
    # test coverage
    h = d = o = n = 0
    for item in test:
        if len(choices_of(item["question"])) == 2:
            o += 1
            continue
        named = named_pages(item["question"], item["docs"])
        if not named:
            n += 1
            continue
        _, head = phrase_and_head(item["question"], named[0]["title"])
        if head:
            h += 1
        else:
            d += 1
    print("test order", o, "head", h, "direct", d, "none", n)
