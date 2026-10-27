"""Development comparison. Unfired questions score 0, so this is not the gated test score."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRUSTMEM_DEVICE", "cuda:1")
os.environ.setdefault("TRUSTMEM_FRACTION", "0.08")

from denote import denote
from message import carry, deliver
from run_durable import load_slices
from run_sota import commit_scored, load_reader, mean, page_rows
from run_trustmem import token_f1


def read_scored(answer, question, doc):
    if not doc:
        return "", 0.0
    return commit_scored(answer, question, page_rows([doc], n=6))[::-1]


def open_read(answer, question, docs):
    best = (0.0, "")
    for doc in docs:
        conf, span = commit_scored(answer, question, page_rows([doc], n=6))
        if conf > best[0]:
            best = (conf, span)
    return best[1]


def main():
    answer = load_reader(os.environ["TRUSTMEM_DEVICE"])

    def read(question, doc):
        return read_scored(answer, question, doc)

    read.score_span = answer.score_span
    for dataset in ("2wiki", "hotpot"):
        dev, _, _ = load_slices(dataset, 120, 4)
        scores = {name: [] for name in ("message", "denote", "carry", "open")}
        fired = exact = 0
        for item in dev:
            question, gold, docs = item["question"], item["answer"], item["docs"]
            qid = item["id"]
            value, kind, _pages, _conf = deliver(question, docs, read, qid)
            old, old_kind, _, _ = denote(question, docs, read)
            carried, _ = carry(question, docs, read, qid)
            opened = open_read(answer, question, docs)
            scores["message"].append(token_f1(value, gold))
            scores["denote"].append(token_f1(old, gold))
            scores["carry"].append(token_f1(carried, gold))
            scores["open"].append(token_f1(opened, gold))
            if kind:
                fired += 1
                if token_f1(value, gold) == 1.0:
                    exact += 1
            if len(scores["message"]) % 12 == 0:
                print(dataset, "seen", len(scores["message"]), flush=True)
        print(
            dataset,
            "n", len(dev),
            "fired", fired,
            "exact", exact,
            "message", round(mean(scores["message"]), 3),
            "denote", round(mean(scores["denote"]), 3),
            "carry", round(mean(scores["carry"]), 3),
            "open", round(mean(scores["open"]), 3),
            flush=True,
        )


if __name__ == "__main__":
    main()
