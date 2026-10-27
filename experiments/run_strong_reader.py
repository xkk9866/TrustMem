"""Same protocol with a stronger reader. Admission rules are unchanged."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_trustmem import (
    FULL_GATES,
    HONEST_CACHE,
    evaluate_split,
    fmt_table,
    load_examples,
    load_models,
)

ROOT = Path(__file__).resolve().parents[1]


def attach_qwen(models, device: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=["roberta", "qwen"], required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-test", type=int, default=120)
    parser.add_argument("--seeds", default="0")
    args = parser.parse_args()
    models = load_models(args.device)
    if args.reader == "roberta":
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        name = "deepset/roberta-large-squad2"
        models.qa_tok = AutoTokenizer.from_pretrained(name)
        models.qa = AutoModelForQuestionAnswering.from_pretrained(name).to(args.device).eval()
    else:
        attach_qwen(models, args.device)
    cfg = {"theta_entail": 0.20, "theta_app": 0.12, "theta_q": 0.15, "margin": 0.20, "entail": True}
    HONEST_CACHE.clear()
    _, test = load_examples("hotpot", 36, args.n_test, seed=3152)
    result = evaluate_split(
        models,
        test,
        cfg,
        seeds=[int(part) for part in args.seeds.split(",")],
        settings=["clean", "mixed", "unsupported", "conflict"],
        methods=["marag", "crag", "trustmem"],
        gates=FULL_GATES,
    )
    print(fmt_table(result["mean"]), flush=True)
    path = ROOT / "results" / f"reader_{args.reader}.json"
    path.write_text(json.dumps({"reader": args.reader, "mean": result["mean"]}, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
