"""Each specialist reads its own pages and writes a verbatim quote.

A quote that is not a sentence on the writer's pages is dropped. When both
specialists answer, keep the answer that the other specialist's pages also
contain. This is the shared-memory rule. Qwen2.5-3B is the specialist.
"""

import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.24, 0)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from run_durable import load_slices, partition
from run_trustmem import normalize_answer, token_f1

DEVICE = "cuda:0"
NAME = "Qwen/Qwen2.5-3B-Instruct"


def page_text(docs, word_cap=900):
    lines = []
    words = 0
    for doc in docs:
        lines.append(doc["title"])
        for sent in doc["sentences"]:
            n = len(sent.split())
            if words + n > word_cap:
                return "\n".join(lines)
            lines.append(sent)
            words += n
    return "\n".join(lines)


def grounded(quote, docs):
    raw = " ".join(" ".join(doc["sentences"]) for doc in docs)
    quote = " ".join(quote.split())
    if len(quote) < 12:
        return False
    return quote in raw or any(quote in sent or sent in quote for doc in docs for sent in doc["sentences"])


def ask(tok, model, question, docs):
    context = page_text(docs)
    if not context.strip():
        return ""
    messages = [
        {
            "role": "system",
            "content": "Reply with only the answer phrase, at most six words. Do not write a sentence.",
        },
        {"role": "user", "content": f"Question: {question}\nPages:\n{context}\nAnswer:"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip().split("\n")[0][:80]


def answer_on_pages(answer, docs):
    needle = normalize_answer(answer)
    if not needle or needle in {"yes", "no"}:
        return False
    blob = normalize_answer(" ".join(" ".join(doc["sentences"]) for doc in docs))
    return needle in blob


def choose(question, proposals):
    kept = [item for item in proposals if item["ok"]]
    if not kept:
        return ""
    q_words = set(normalize_answer(question).split())

    def page_overlap(item):
        return len(q_words & set(normalize_answer(item["blob"]).split()))

    witnessed = []
    for item in kept:
        needle = normalize_answer(item["answer"])
        if not needle or needle in q_words:
            continue
        others = " ".join(other["blob"] for other in proposals if other is not item)
        if needle in normalize_answer(others):
            witnessed.append(item)
    pool = witnessed or kept
    return max(pool, key=page_overlap)["answer"]


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def run_split(tok, model, examples):
    scores, alone, bests = [], [], []
    for index, ex in enumerate(examples):
        rng = random.Random(f"{ex['id']}:2:random")
        groups = partition(ex["docs"], 2, "random", rng)
        proposals = []
        for docs in groups:
            answer = ask(tok, model, ex["question"], docs)
            proposals.append(
                {
                    "answer": answer,
                    "ok": answer_on_pages(answer, docs),
                    "blob": page_text(docs, word_cap=500),
                }
            )
        pred = choose(ex["question"], proposals)
        grounded_f1 = [token_f1(item["answer"], ex["answer"]) for item in proposals if item["ok"]]
        either = max(grounded_f1) if grounded_f1 else 0.0
        all_answer = ask(tok, model, ex["question"], ex["docs"])
        if not answer_on_pages(all_answer, ex["docs"]):
            all_answer = ""
        scores.append(token_f1(pred, ex["answer"]))
        alone.append(token_f1(all_answer, ex["answer"]))
        bests.append(either)
        if (index + 1) % 20 == 0:
            print(
                f"  {index + 1} split {np.mean(scores):.3f}"
                f" either {np.mean(bests):.3f} all {np.mean(alone):.3f}",
                flush=True,
            )
    return scores, alone, bests


def main():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(DEVICE).eval()
    _, test, _ = load_slices("hotpot", 120, 4)
    print("hotpot", len(test), flush=True)
    scores, alone, bests = run_split(tok, model, test)
    print(
        f"split {np.mean(scores):.3f}  either {np.mean(bests):.3f}  all-pages {np.mean(alone):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
