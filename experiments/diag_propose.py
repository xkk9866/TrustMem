"""Print three raw specialist replies so a bad parse can be separated from a bad answer."""

import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.24, 0)

from run_durable import load_slices, partition

DEVICE = "cuda:0"
NAME = "Qwen/Qwen2.5-3B-Instruct"


def page_text(docs, word_cap=700):
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


def main():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(DEVICE).eval()
    _, test, _ = load_slices("hotpot", 3, 1)
    for ex in test:
        rng = random.Random(f"{ex['id']}:2:random")
        groups = partition(ex["docs"], 2, "random", rng)
        print("\nQ:", ex["question"])
        print("A:", ex["answer"])
        context = page_text(groups[0])
        messages = [
            {
                "role": "system",
                "content": "Answer with a short fact drawn from the pages. Do not explain.",
            },
            {"role": "user", "content": f"Question: {ex['question']}\nPages:\n{context}\nAnswer:"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800).to(DEVICE)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print("RAW:", repr(text[:300]))
        print("PAGES:", context[:240].replace("\n", " | "))


if __name__ == "__main__":
    main()
