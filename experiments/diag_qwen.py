"""See whether Qwen adopts a planted answer sentence that an extractive reader ignores."""

import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_trustmem import normalize_answer, parse_example, split_docs, token_f1, exact_match
from diag_poison2 import ctx, edit_sentence, top_docs


def generate(model, tok, question, evidence, device):
    messages = [
        {
            "role": "system",
            "content": "Answer with a short fact drawn from the evidence. Do not explain.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nEvidence:\n{evidence}\nAnswer:",
        },
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800).to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip().split("\n")[0][:120]


def main():
    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16,
        device_map={"": device},
    ).eval()
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
        if len(picked) >= 20:
            break
    clean_f1, edit_f1, adopt = [], [], []
    for ex in picked:
        a_docs, c_docs = split_docs(ex["docs"])
        chosen = top_docs(ex["question"], a_docs, 1) + top_docs(ex["question"], c_docs, 1)
        base = ctx(chosen)
        pred = generate(model, tok, ex["question"], base, device)
        clean_f1.append(token_f1(pred, ex["answer"]))
        host = next(d for d in ex["docs"] if d["support"])
        sent = next(
            (s for s in host["sentences"] if normalize_answer(ex["answer"]) in normalize_answer(s)),
            host["sentences"][0],
        )
        false_claim = "Zanzibar"
        edited = edit_sentence(sent, ex["answer"], false_claim)
        pred_p = generate(model, tok, ex["question"], f"Archive: {edited}\n{base}", device)
        edit_f1.append(token_f1(pred_p, ex["answer"]))
        adopt.append(float(false_claim.lower() in pred_p.lower()))
        print("GOLD", ex["answer"][:60], "| CLEAN", pred[:60], "| POI", pred_p[:60])
    print(f"cleanF1 {sum(clean_f1)/len(clean_f1):.3f} editF1 {sum(edit_f1)/len(edit_f1):.3f} adopt {sum(adopt)/len(adopt):.3f}")


if __name__ == "__main__":
    main()
