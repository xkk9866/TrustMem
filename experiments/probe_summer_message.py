"""Remeasure the Summer's Blood test question under owner-only delivery."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRUSTMEM_DEVICE", "cuda:1")
os.environ.setdefault("TRUSTMEM_FRACTION", "0.08")

from message import deliver
from phrase import candidates
from run_durable import load_slices
from run_message import read_scored
from run_sota import DEVICE, load_reader

reader = load_reader(DEVICE)

def read(question, doc):
    return read_scored(reader, question, doc)

read.score_span = reader.score_span
_, test, _ = load_slices("2wiki", 120, 4)
found = False
for item in test:
    question = item["question"]
    if "Summer" not in question or "Blood" not in question:
        continue
    found = True
    print("question", question)
    print("gold", item["answer"])
    for row in candidates(question, item["docs"])[:4]:
        print("candidate", row["page"], "|", row["head"], "|", row["phrase"])
    value, kind, pages, conf = deliver(question, item["docs"], read, item["id"])
    print("deliver", repr(value), kind, pages, round(conf, 4))
if not found:
    print("not in test slice")
