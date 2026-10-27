"""Same open-retrieval methods with Qwen2.5-3B as the reader. Hotpot test only."""

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_openrag import answer_method, load_reader, paired_effect, summarize_scores
from run_trustmem import load_examples, token_f1

ROOT_OUT = __import__("pathlib").Path(__file__).resolve().parents[1] / "results" / "openrag_qwen.json"


def main() -> None:
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    models = load_reader(device)
    name = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()

    def answer(question: str, context: str) -> tuple[str, float]:
        if not context.strip():
            return "", 0.0
        messages = [
            {"role": "system", "content": "Answer with a short fact drawn from the evidence. Do not explain."},
            {"role": "user", "content": f"Question: {question}\nEvidence:\n{context}\nAnswer:"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return text.strip().split("\n")[0][:80], 0.55

    models.answer = answer
    _, test = load_examples("hotpot", 36, 240, seed=3152)
    methods = ["bm25", "rrf5", "bm25link", "agree", "trust", "goldpages"]
    scores = {method: [] for method in methods}
    support = {method: [] for method in methods}
    for index, ex in enumerate(test):
        for method in methods:
            out = answer_method(models, method, ex["question"], ex["docs"])
            scores[method].append(token_f1(out["pred"], ex["answer"]))
            support[method].append(out["support"])
        if (index + 1) % 20 == 0:
            print(index + 1, flush=True)
    rng = np.random.default_rng(3152)
    block = {"reader": name, "n": len(test), "methods": {}}
    for method in methods:
        stats = summarize_scores(scores[method], rng)
        stats["support"] = float(np.mean(support[method]))
        block["methods"][method] = stats
        print(f"{method:10s} {stats['f1']:.3f} [{stats['lo']:.3f}, {stats['hi']:.3f}]", flush=True)
    trust = np.asarray(scores["trust"])
    block["paired"] = {
        f"trust_minus_{method}": paired_effect(trust, np.asarray(scores[method]), rng)
        for method in methods
        if method != "trust"
    }
    ROOT_OUT.write_text(json.dumps(block, indent=2), encoding="utf-8")
    print("wrote", ROOT_OUT, flush=True)


if __name__ == "__main__":
    main()
