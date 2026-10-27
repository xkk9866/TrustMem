"""Same durable packets, Qwen2.5-3B as the shared reader.

Compares the corrective four sentences with the attested log. Every method
uses this reader. Numbers from the extractive reader are a different readout.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.24, 0)

from run_trustmem import token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"
NAME = "Qwen/Qwen2.5-3B-Instruct"


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            title = ev.get("title") or ""
            line = f"{title}: {text}".strip() if title else text
            if text and line not in seen:
                seen.add(line)
                out.append(line)
    return out


def context_of(question, lines, limit=12):
    q = tokens(question)
    ranked = sorted(lines, key=lambda line: len(q & tokens(line)), reverse=True)
    kept = []
    words = 0
    for line in ranked[:limit]:
        n = len(line.split())
        if kept and words + n > 420:
            break
        kept.append(line)
        words += n
    return "\n".join(kept)


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float16).to(DEVICE).eval()

    def generate(question, context):
        if not context.strip():
            return ""
        messages = [
            {"role": "system", "content": "Answer with a short fact drawn from the evidence. Do not explain."},
            {"role": "user", "content": f"Question: {question}\nEvidence:\n{context}\nAnswer:"},
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

    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "clean"):
            scores = {name: [] for name in ("crag_plus", "trustmem")}
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                for method in ("crag_plus", "trustmem"):
                    ctx = context_of(question, rows(packets[setting][method][index]))
                    scores[method].append(token_f1(generate(question, ctx), gold))
            print(
                f"  {setting:12s} plus {np.mean(scores['crag_plus']):.3f}"
                f"  mem {np.mean(scores['trustmem']):.3f}",
                flush=True,
            )
            delta, lo, hi = bootstrap(scores["trustmem"], scores["crag_plus"])
            print(f"    mem-plus: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
