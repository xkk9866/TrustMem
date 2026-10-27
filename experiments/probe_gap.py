"""Where the message protocol and the previous reduction disagree on the development prefix."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRUSTMEM_DEVICE", "cuda:1")
os.environ.setdefault("TRUSTMEM_FRACTION", "0.08")

from denote import denote
from message import deliver
from run_durable import load_slices
from run_sota import commit_scored, load_reader, page_rows
from run_trustmem import token_f1


def read_scored(answer, question, doc):
    if not doc:
        return "", 0.0
    return commit_scored(answer, question, page_rows([doc], n=6))[::-1]


def main():
    answer = load_reader(os.environ["TRUSTMEM_DEVICE"])

    def read(question, doc):
        return read_scored(answer, question, doc)

    read.score_span = answer.score_span
    dev, _, _ = load_slices("2wiki", 120, 4)
    for item in dev:
        question, gold, docs = item["question"], item["answer"], item["docs"]
        value, kind, pages, conf = deliver(question, docs, read, item["id"])
        old, old_kind, _, old_conf = denote(question, docs, read)
        new_f1, old_f1 = token_f1(value, gold), token_f1(old, gold)
        if old_f1 - new_f1 > 0.2 or new_f1 - old_f1 > 0.2:
            print(f"new {new_f1:.2f} {value!r} old {old_f1:.2f} {old!r}")
            print("  ", question)
            print("  gold", gold, "kind", kind, old_kind, "pages", pages)


if __name__ == "__main__":
    main()
