"""Accuracy of the order path on the test slice."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from denote import denote
from diag_compare2 import choices_of
from run_durable import load_slices
from run_sota import commit_scored, load_reader, page_rows
from run_trustmem import normalize_answer, token_f1

reader = load_reader(os.environ.get("TRUSTMEM_DEVICE", "cuda:1"))


def read(question, doc):
    conf, span = commit_scored(reader, question, page_rows([doc], n=6))
    return span, conf


read.score_span = reader.score_span

for dataset in ("2wiki", "hotpot"):
    _, test, _ = load_slices(dataset, 120, 4)
    fired = exact = partial = 0
    scores = []
    wrong = []
    for item in test:
        if len(choices_of(item["question"])) != 2:
            continue
        value, kind, _, _ = denote(item["question"], item["docs"], read)
        if kind != "order":
            continue
        fired += 1
        score = token_f1(value, item["answer"])
        scores.append(score)
        if normalize_answer(value) == normalize_answer(item["answer"]):
            exact += 1
        elif score < 0.5:
            wrong.append((item["question"][:80], value, item["answer"]))
        else:
            partial += 1
    mean = sum(scores) / len(scores) if scores else 0
    print(dataset, "fired", fired, "exact", exact, "partial", partial, "mean", round(mean, 3), flush=True)
    for row in wrong[:6]:
        print(" ", row, flush=True)
