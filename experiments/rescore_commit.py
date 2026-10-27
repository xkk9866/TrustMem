"""2Wiki and the remaining Hotpot cells for single-sentence commitment."""

import gc
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.18, 0)

from run_trustmem import normalize_answer, token_f1

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"
QA_TOK = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-squad2")
QA = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-base-squad2").to(DEVICE).eval()


def solo(question, context):
    enc = QA_TOK(
        question,
        context,
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
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
    a, b = offsets[best[1]][0], offsets[best[2]][1]
    if a is None or b is None or b <= a:
        return "", 0.0
    return context[a:b].strip(), best[0]


def rows(stored):
    out, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(ev)
    return out


def propose(question, evs, penalize_title: bool):
    titles = {normalize_answer((ev.get("title") or "").split("(")[0]) for ev in evs}
    best = (0.0, "")
    for ev in evs:
        line = f"{ev.get('title', '')}: {ev.get('text', '')}".strip()
        pred, conf = solo(question, line)
        needle = normalize_answer(pred)
        if penalize_title and needle and any(
            needle == title or (len(needle) > 3 and needle in title) for title in titles if title
        ):
            conf *= 0.25
        if conf > best[0]:
            best = (conf, pred)
    return best[1]


def main():
    jobs = (
        ("2wiki", ("mixed", "unsupported", "incomplete", "clean", "conflict")),
        ("hotpot", ("conflict",)),
    )
    for dataset, settings in jobs:
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in settings:
            plain, penal = [], []
            for item, index in zip(meta, range(len(meta))):
                evs = rows(packets[setting]["crag_plus"][index])
                question = item["question"]
                gold = item["answer"]
                plain.append(token_f1(propose(question, evs, False), gold))
                penal.append(token_f1(propose(question, evs, True), gold))
            print(
                f"  {setting:12s} plain {float(np.mean(plain)):.3f}  title-penalty {float(np.mean(penal)):.3f}",
                flush=True,
            )
            del plain, penal
            gc.collect()


if __name__ == "__main__":
    main()
