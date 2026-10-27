"""Two channels propose one sentence each. Keep the more committed proposal.

The corrective channel is the page read. The attested channel is the long log.
Each channel answers from its own most committed sentence. Shared memory
returns the proposal with the higher reader commitment.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.18, 0)

from run_trustmem import normalize_answer, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"
QA_TOK = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-squad2")
QA = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-base-squad2").to(DEVICE).eval()
EMB_TOK = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
EMB = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEVICE).eval()
CACHE = {}


def embed(text, query):
    raw = text.strip()[:800]
    key = (query, raw)
    if key in CACHE:
        return CACHE[key]
    payload = ("Represent this sentence for searching relevant passages: " + raw) if query else raw
    enc = EMB_TOK(payload, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        vec = EMB(**enc).last_hidden_state[0, 0].detach().cpu().numpy().astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    CACHE[key] = vec
    return vec


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


def rank_top(question, evs, k=4):
    q_toks = tokens(question)
    qvec = embed(question, True)
    scored = []
    for ev in evs:
        text = ev.get("text") or ""
        overlap = (len(q_toks & tokens(text)) / len(q_toks)) if q_toks else 0.0
        cos = float(np.dot(qvec, embed(text, False)))
        scored.append((0.45 * overlap + 0.55 * max(cos, 0.0), ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [ev for _, ev in scored[:k]]


def best_proposal(question, evs):
    titles = {normalize_answer((ev.get("title") or "").split("(")[0]) for ev in evs}
    best = (0.0, "")
    for ev in evs:
        line = f"{ev.get('title', '')}: {ev.get('text', '')}".strip()
        pred, conf = solo(question, line)
        needle = normalize_answer(pred)
        if needle and any(needle == title or (len(needle) > 3 and needle in title) for title in titles if title):
            conf *= 0.25
        if conf > best[0]:
            best = (conf, pred)
    return best


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "clean", "conflict"):
            if setting not in packets:
                continue
            scores = {name: [] for name in ("plus", "mem", "channel")}
            took_mem = 0
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                plus = rank_top(question, rows(packets[setting]["crag_plus"][index]))
                mem = rank_top(question, rows(packets[setting]["trustmem"][index]))
                conf_p, pred_p = best_proposal(question, plus)
                conf_m, pred_m = best_proposal(question, mem)
                if conf_m > conf_p:
                    pred = pred_m
                    took_mem += 1
                else:
                    pred = pred_p
                scores["plus"].append(token_f1(pred_p, gold))
                scores["mem"].append(token_f1(pred_m, gold))
                scores["channel"].append(token_f1(pred, gold))
            print(
                f"  {setting:12s} plus {np.mean(scores['plus']):.3f}"
                f"  mem {np.mean(scores['mem']):.3f}"
                f"  channel {np.mean(scores['channel']):.3f}"
                f"  mem_chosen {took_mem}/{len(meta)}",
                flush=True,
            )
            for base in ("plus", "mem"):
                delta, lo, hi = bootstrap(scores["channel"], scores[base])
                print(f"    channel-{base}: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
