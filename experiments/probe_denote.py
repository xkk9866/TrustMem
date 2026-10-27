"""Development-slice check for denotation transfer. Not a test score."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from denote import containing_phrase, denote, head_word, inner_question
from diag_compare2 import choices_of
from run_durable import load_slices
from run_sota import commit_scored, load_reader, page_rows
from run_trustmem import token_f1

reader = load_reader(os.environ.get("TRUSTMEM_DEVICE", "cuda:1"))


def read(question, doc):
    conf, span = commit_scored(reader, question, page_rows([doc], n=6))
    return span, conf


read.score_span = reader.score_span


def open_all(question, docs):
    best = (0.0, "")
    for doc in docs:
        span, conf = read(question, doc)
        if conf > best[0]:
            best = (conf, span)
    return best[1]


for dataset in ("2wiki", "hotpot"):
    dev, _, _ = load_slices(dataset, 120, 4)
    scores = {"denote": [], "open": [], "order": [], "bridge": []}
    fired = 0
    misses = []
    for item in dev:
        value, kind, pages, conf = denote(item["question"], item["docs"], read)
        pred = value if kind else ""
        if kind:
            fired += 1
        gold = item["answer"]
        dscore = token_f1(pred, gold) if kind else 0.0
        oscore = token_f1(open_all(item["question"], item["docs"]), gold)
        scores["denote"].append(dscore)
        scores["open"].append(oscore)
        bucket = "order" if len(choices_of(item["question"])) == 2 else "bridge"
        scores[bucket].append(dscore)
        if kind and dscore < 0.5:
            misses.append((round(dscore, 2), kind, item["question"][:90], pred, gold))
    n = len(dev)
    print(
        dataset,
        "n", n,
        "fired", fired,
        "denote", round(sum(scores["denote"]) / n, 3),
        "open", round(sum(scores["open"]) / n, 3),
        "order", round(sum(scores["order"]) / max(len(scores["order"]), 1), 3),
        "norder", len(scores["order"]),
        "bridge", round(sum(scores["bridge"]) / max(len(scores["bridge"]), 1), 3),
        flush=True,
    )
    for row in misses[:8]:
        print("  miss", row, flush=True)

dev, test, _ = load_slices("2wiki", 120, 4)
for item in list(dev) + list(test):
    if "Summer" not in item["question"] or "Blood" not in item["question"]:
        continue
    named = [doc for doc in item["docs"] if doc["title"].lower() in item["question"].lower()]
    named.sort(key=lambda doc: len(doc["title"]), reverse=True)
    phrase = containing_phrase(item["question"], named[0]["title"])
    print("SUMMER", item["question"])
    print(" phrase", phrase, "head", head_word(phrase, named[0]["title"]))
    print(" inner", inner_question(phrase, named[0]["title"], head_word(phrase, named[0]["title"])))
    print(" denote", denote(item["question"], item["docs"], read))
    break
