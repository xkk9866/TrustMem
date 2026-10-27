"""Does an entailment score put the answer sentence ahead of the biography? CPU only."""

import pickle
from pathlib import Path

from run_trustmem import load_models, normalize_answer

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cpu")


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def main():
    with (ROOT / "results" / "packets_hotpot.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    shown = 0
    for index, item in enumerate(meta):
        gold = item["answer"]
        needle = normalize_answer(gold)
        mem = rows(packets["unsupported"]["trustmem"][index])
        if not any(needle and needle in normalize_answer(text) for text in mem):
            continue
        question = item["question"]
        hyp = f"This passage answers the question: {question}"
        ranked = []
        for text in mem:
            ranked.append((MODELS.entail_prob(text, hyp), needle in normalize_answer(text), text))
        ranked.sort(reverse=True)
        print("\nQ:", question)
        print("A:", gold)
        for score, hit, text in ranked[:5]:
            mark = "*" if hit else " "
            print(f"  {mark} {score:.3f} {text[:120]}")
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()
