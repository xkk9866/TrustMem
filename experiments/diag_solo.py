"""Solo reader output on unsupported sentences, to see what separates the answer sentence."""

import pickle
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.15, 1)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from run_trustmem import normalize_answer, token_f1

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:1"
TOK = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-squad2")
QA = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-base-squad2").to(DEVICE).eval()


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


def solo(question, context):
    enc = TOK(question, context, truncation="only_second", max_length=512, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids(0)
    batch = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        out = QA(**batch)
    start = torch.softmax(out.start_logits[0], dim=-1).detach().cpu().numpy()
    end = torch.softmax(out.end_logits[0], dim=-1).detach().cpu().numpy()
    best = None
    for i, sid in enumerate(seq_ids):
        if sid != 1:
            continue
        for j in range(i, min(i + 24, len(seq_ids))):
            if seq_ids[j] != 1:
                break
            score = float(start[i] * end[j])
            if best is None or score > best[0]:
                best = (score, i, j)
    if best is None:
        return "", 0.0
    span = context[offsets[best[1]][0] : offsets[best[2]][1]].strip()
    return span, best[0]


def main():
    with (ROOT / "results" / "packets_hotpot.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    shown = 0
    for index, item in enumerate(meta):
        gold = item["answer"]
        needle = normalize_answer(gold)
        mem = rows(packets["unsupported"]["trustmem"][index])
        if not any(needle and needle in normalize_answer(ev.get("text") or "") for ev in mem):
            continue
        print("\nQ:", item["question"])
        print("A:", gold)
        for ev in mem:
            text = ev.get("text") or ""
            pred, conf = solo(item["question"], f"{ev.get('title', '')}: {text}")
            mark = "*" if needle in normalize_answer(text) else " "
            print(
                f"  {mark} {conf:6.3f} pred={pred[:40]!r:42} f1={token_f1(pred, gold):.2f} {text[:110]}"
            )
        shown += 1
        if shown >= 6:
            break


if __name__ == "__main__":
    main()
