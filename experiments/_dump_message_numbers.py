"""Format message_read.json and count question shapes. No torch."""
import json
import random
import re
import string
from pathlib import Path

from datasets import load_dataset

from phrase import candidates

ROOT = Path(__file__).resolve().parents[1]


def f2(value):
    return format(value, ".2f")


def f3(value):
    return format(value, ".3f")


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
    for title, sentences in zip(titles, sent_groups):
        clean = [re.sub(r"\s+", " ", str(sent)).strip() for sent in sentences]
        docs.append({
            "title": title,
            "sentences": [text for text in clean if text],
            "support": title in support_titles,
        })
    return {
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type") or "unknown",
        "docs": docs,
        "id": ex.get("id") or ex.get("_id") or "",
    }


def answer_supported(item):
    blob = normalize_answer(" ".join(sent for doc in item["docs"] for sent in doc["sentences"]))
    ans = normalize_answer(item["answer"])
    return bool(ans) and ans in blob and len(ans.split()) <= 12


def choices_of(question):
    text = question.strip().rstrip("?")
    match = re.search(r",\s*(.+?)\s+or\s+(.+)$", text, flags=re.I)
    return [match.group(1).strip(), match.group(2).strip()] if match else []


def load(dataset):
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
    return parsed[:36], parsed[36:156]


report = json.loads((ROOT / "results" / "message_read.json").read_text(encoding="utf-8"))
seed = json.loads((ROOT / "results" / "message_read_seed.json").read_text(encoding="utf-8"))
streams = ["mixed", "unsupported", "incomplete", "clean", "conflict"]
names = ["marag", "crag", "faithful", "open", "carry", "chain", "gated"]

for dataset in ("2wiki", "hotpot"):
    block = report["datasets"][dataset]
    print("\n==", dataset, "orders", block["orders"])
    for stream in streams:
        row = block["settings"][stream]
        print(stream, row["fired"])
        for name in names:
            print(f"  {name:10s} {f2(row[name])}  {row[name]:.6f}")
        for key, delta in row["delta"].items():
            star = " *" if delta["lo"] > 0 else ""
            below = " BELOW" if delta["hi"] < 0 else ""
            print(
                f"  {key:20s} {f3(delta['mean'])} [{f3(delta['lo'])}, {f3(delta['hi'])}]"
                f"{star}{below} lo={delta['lo']:.8f}"
            )
        buckets = {}
        for name in names:
            buckets.setdefault(f2(row[name]), []).append(name)
        clashes = {key: value for key, value in buckets.items() if len(value) > 1}
        if clashes:
            print("  COLLIDE", clashes)

for dataset in ("2wiki", "hotpot"):
    src = seed["datasets"][dataset]["seed2027"]
    print("\nSEED", dataset)
    for name in ("message", "open", "carry"):
        print(f"  {name:10s} {f2(src[name])}  {src[name]:.6f}")
    delta = src["delta"]["message-open"]
    print(
        f"  message-open {f3(delta['mean'])} [{f3(delta['lo'])}, {f3(delta['hi'])}] "
        f"lo={delta['lo']:.8f} hi={delta['hi']:.8f}"
    )
    report["datasets"][dataset]["seed2027"] = {
        "protocol": (
            "Ungated binding on parsed[36:156] after shuffle seed 2027. "
            "No packets. An empty binding scores zero. "
            "This block keeps message and open only."
        ),
        "message": src["message"],
        "open": src["open"],
        "delta": {"message-open": src["delta"]["message-open"]},
    }

(ROOT / "results" / "message_read.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print("\nmerged seed2027 message/open into message_read.json")

for dataset in ("2wiki", "hotpot"):
    dev, test = load(dataset)
    print("\nCENSUS", dataset)
    for split, rows in (("dev", dev), ("test", test)):
        counts = {"order": 0, "headed": 0, "direct": 0, "none": 0}
        summer = []
        for item in rows:
            question = item["question"]
            if "Summer" in question and "Blood" in question:
                summer.append(question)
            if len(choices_of(question)) == 2:
                counts["order"] += 1
                continue
            rows_c = candidates(question, item["docs"])
            if not rows_c:
                counts["none"] += 1
            elif rows_c[0]["head"]:
                counts["headed"] += 1
            else:
                counts["direct"] += 1
        print(split, len(rows), counts, "summer", summer)
