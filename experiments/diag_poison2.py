"""Does an edited support sentence pull the extractive reader off the gold span?"""

import random
import re

import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from run_trustmem import exact_match, load_models, normalize_answer, parse_example, split_docs, token_f1


def top_docs(question, docs, k):
    tokenized = [re.findall(r"[a-z0-9]+", " ".join(d["sentences"]).lower()) or ["empty"] for d in docs]
    ranker = BM25Okapi(tokenized)
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    scores = ranker.get_scores(query)
    order = np.argsort(-np.asarray(scores))[:k]
    return [docs[int(i)] for i in order]


def ctx(docs):
    return "\n".join(f"{d['title']}: {s}" for d in docs for s in d["sentences"][:4])


def edit_sentence(text, answer, false_claim):
    norm_text = normalize_answer(text)
    norm_ans = normalize_answer(answer)
    if norm_ans and norm_ans in norm_text:
        # Replace the first case-insensitive occurrence of the raw answer if present,
        # otherwise splice the false claim in front of the sentence.
        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(false_claim, text, count=1)
    return f"{false_claim} is the answer. {text}"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    raw = load_dataset("hotpot_qa", "distractor", split="validation")
    rng = random.Random(3152)
    idxs = list(range(len(raw)))
    rng.shuffle(idxs)
    picked = []
    for i in idxs:
        ex = parse_example(raw[int(i)])
        if ex["type"] != "bridge":
            continue
        blob = " ".join(s for d in ex["docs"] for s in d["sentences"])
        if normalize_answer(ex["answer"]) not in normalize_answer(blob):
            continue
        picked.append(ex)
        if len(picked) >= 40:
            break

    stats = {k: [] for k in ["clean_f1", "clean_em", "edit_f1", "edit_em", "adopt"]}
    for ex in picked:
        a_docs, c_docs = split_docs(ex["docs"])
        chosen = top_docs(ex["question"], a_docs, 1) + top_docs(ex["question"], c_docs, 1)
        base = ctx(chosen)
        pred, _ = models.answer(ex["question"], base)
        stats["clean_f1"].append(token_f1(pred, ex["answer"]))
        stats["clean_em"].append(exact_match(pred, ex["answer"]))
        host = next(d for d in ex["docs"] if d["support"])
        sent = next(
            (s for s in host["sentences"] if normalize_answer(ex["answer"]) in normalize_answer(s)),
            host["sentences"][0],
        )
        false_claim = "Zanzibar"
        edited = edit_sentence(sent, ex["answer"], false_claim)
        pred_p, _ = models.answer(ex["question"], f"Archive: {edited}\n{base}")
        stats["edit_f1"].append(token_f1(pred_p, ex["answer"]))
        stats["edit_em"].append(exact_match(pred_p, ex["answer"]))
        stats["adopt"].append(exact_match(pred_p, false_claim))
    for key, vals in stats.items():
        print(f"{key:10s} {sum(vals)/len(vals):.3f}")


if __name__ == "__main__":
    main()
