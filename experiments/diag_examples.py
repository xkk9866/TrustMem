"""Score the two overview examples with the shipped operators."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from diag_compare2 import pick
from run_durable import load_slices
from run_sota import load_reader, read_page
from witness_memory import express

_, test, _ = load_slices("2wiki", 120, 4)
reader = load_reader(os.environ.get("TRUSTMEM_DEVICE", "cuda:1"))
for item in test:
    question = item["question"]
    if "Gallivant" not in question and "Summer" not in question:
        continue
    print("Q:", question)
    print("gold:", item["answer"])
    print("pick:", pick(question, item["docs"]))
    value, how, pages = express(
        question,
        item["docs"],
        lambda asked, doc: read_page(reader, asked, doc),
    )
    print("express:", value, how, [page["title"] for page in pages])
