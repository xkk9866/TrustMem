"""Run the comparison operator on two test questions without loading a reader."""
import importlib.util
import random
import re
import string
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


stub = types.ModuleType("run_trustmem")
stub.normalize_answer = normalize_answer
stub.token_f1 = lambda *args, **kwargs: 0.0
sys.modules["run_trustmem"] = stub
durable = types.ModuleType("run_durable")
durable.load_slices = lambda *args, **kwargs: None
sys.modules["run_durable"] = durable

spec = importlib.util.spec_from_file_location(
    "diag_compare2", ROOT / "experiments" / "diag_compare2.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from datasets import load_dataset

raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
print("keys", raw[0].keys())


def to_docs(example):
    context = example.get("context") or example.get("contexts")
    docs = []
    if isinstance(context, dict) and "title" in context:
        titles = context["title"]
        sentences = context["content"] if "content" in context else context["sentences"]
        for title, sents in zip(titles, sentences):
            docs.append({"title": title, "sentences": list(sents)})
        return docs
    if isinstance(context, list):
        for block in context:
            if isinstance(block, dict):
                docs.append(
                    {
                        "title": block.get("title", ""),
                        "sentences": list(block.get("sentences") or block.get("content") or []),
                    }
                )
            elif isinstance(block, (list, tuple)) and len(block) == 2:
                docs.append({"title": block[0], "sentences": list(block[1])})
    return docs


wanted = []
for example in raw:
    question = example["question"]
    if "Gallivant" in question or "Summer" in question or "Deadly Eyes" in question:
        wanted.append(example)

print("found", len(wanted))
for example in wanted:
    docs = to_docs(example)
    print("---")
    print(example["question"])
    print("gold", example["answer"])
    print("docs", [doc["title"] for doc in docs][:8], "n", len(docs))
    print("pick", mod.pick(example["question"], docs))
    if "Gallivant" in example["question"] and "Divine" in example["question"]:
        for doc in docs:
            if doc["title"] in {"Divine Food", "Gallivant"}:
                print("YEAR", doc["title"], mod.release_year(doc))
                print("S0", doc["sentences"][0][:240] if doc["sentences"] else "")
