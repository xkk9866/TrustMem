"""Headroom inside the CRAG+ packet, and the sentences the reader actually prefers."""

import pickle
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.40, 0)

from run_trustmem import load_models, normalize_answer, token_f1

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models("cuda:0")


def packet(stored):
    rows, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            key = (ev.get("title") or "", ev.get("text") or "")
            if key[1] and key not in seen:
                seen.add(key)
                rows.append(key)
    return rows


def solo(question, title, text):
    pred, score = MODELS.answer(question, f"{title}: {text}")
    return pred, float(score)


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported"):
            chosen_f1, oracle_f1, covered = [], [], []
            shown = 0
            for index, item in enumerate(meta):
                question, gold = item["question"], item["answer"]
                gold_n = normalize_answer(gold)
                rows = []
                for title, text in packet(packets[setting]["crag_plus"][index]):
                    pred, conf = solo(question, title, text)
                    holds = bool(gold_n) and gold_n in normalize_answer(text)
                    rows.append((conf, token_f1(pred, gold), holds, pred, title, text))
                if not rows:
                    chosen_f1.append(0.0)
                    oracle_f1.append(0.0)
                    covered.append(0)
                    continue
                pick = max(rows, key=lambda row: row[0])
                holders = [row for row in rows if row[2]]
                covered.append(int(bool(holders)))
                oracle = max(holders, key=lambda row: row[1])[1] if holders else pick[1]
                chosen_f1.append(pick[1])
                oracle_f1.append(oracle)
                if pick[1] < 0.5 and shown < 6:
                    shown += 1
                    print(f"  [{setting}] gold={gold} pick={pick[3]!r} conf={pick[0]:.3f} f1={pick[1]:.2f}", flush=True)
                    print(f"    Q: {question}", flush=True)
                    for conf, f1, holds, pred, title, text in sorted(rows, key=lambda row: -row[0])[:4]:
                        flag = " GOLD" if holds else ""
                        print(f"    {conf:.3f} f1={f1:.2f}{flag} {title}: {pred!r} | {text[:110]}", flush=True)
            print(
                f"  {setting}: commit {np.mean(chosen_f1):.3f}"
                f"  oracle-sentence {np.mean(oracle_f1):.3f}"
                f"  gold-in-packet {np.mean(covered):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
