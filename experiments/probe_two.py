"""Confirm the two overview examples under the shipped derivation."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from derivation import derive, outer_question
from run_durable import load_slices
from run_sota import commit_scored, load_reader, page_rows

reader = load_reader(os.environ.get("TRUSTMEM_DEVICE", "cuda:1"))


def read(question, doc):
    conf, span = commit_scored(reader, question, page_rows([doc], n=6))
    return span, conf


_, test, _ = load_slices("2wiki", 120, 4)
for item in test:
    question = item["question"]
    if "Gallivant" not in question and "Summer" not in question:
        continue
    value, kind, pages, conf = derive(question, item["docs"], read)
    print("Q:", question)
    print("gold:", item["answer"])
    print("derive:", value, kind, conf, [p["title"] for p in pages])
