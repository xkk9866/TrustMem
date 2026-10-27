"""Print the first and residual reads for the Summer's Blood question."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from derivation import derive, inner_question, outer_question, titles_named
from diag_compare2 import find_doc
from run_durable import load_slices
from run_sota import commit_scored, load_reader, page_rows

reader = load_reader(os.environ.get("TRUSTMEM_DEVICE", "cuda:1"))
_, test, _ = load_slices("2wiki", 120, 4)
for item in test:
    question = item["question"]
    if "Summer" not in question or "Blood" not in question:
        continue
    docs = item["docs"]
    anchor = titles_named(question, docs)[0]
    asked_inner = inner_question(question, anchor["title"])
    print("inner", asked_inner)
    conf, span = commit_scored(reader, asked_inner, page_rows([anchor], n=6))
    print("anchor", anchor["title"])
    print("span", repr(span), "conf", round(conf, 4))
    print("lede", anchor["sentences"][0][:300])
    target = find_doc(docs, span) if span else None
    print("target", None if target is None else target["title"])
    if target:
        asked = outer_question(question, anchor["title"], span)
        print("asked", asked)
        conf2, pred = commit_scored(reader, asked, page_rows([target], n=6))
        print("pred", repr(pred), "conf", round(conf2, 4))
    value, kind, pages, kept = derive(
        question,
        docs,
        lambda q, doc: commit_scored(reader, q, page_rows([doc], n=6))[::-1],
    )
    print("derive", value, kind, kept, [p["title"] for p in pages])
