"""Long-horizon check of single-sentence commitment.

Rank the log by the same applicability score as the paper, keep four sentences,
and answer from the one the reader commits to. Title-only spans are down-weighted.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.20, 0)

from run_trustmem import normalize_answer, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"
QA_TOK = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-squad2")
QA = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-base-squad2").to(DEVICE).eval()
EMB_TOK = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
EMB = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEVICE).eval()
CACHE = {}


def clip(text):
    return (text or "").strip()[:800].strip()


def embed_many(texts):
    missing = []
    for text in texts:
        raw = clip(text)
        if raw and raw not in CACHE:
            missing.append(raw)
    missing = list(dict.fromkeys(missing))
    for start in range(0, len(missing), 64):
        batch = missing[start : start + 64]
        enc = EMB_TOK(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.inference_mode():
            vecs = EMB(**enc).last_hidden_state[:, 0].detach().cpu().numpy().astype(np.float32)
        for raw, vec in zip(batch, vecs):
            CACHE[raw] = vec / (np.linalg.norm(vec) + 1e-8)


def embed_query(question):
    key = "Q:" + question.strip()[:800]
    if key in CACHE:
        return CACHE[key]
    payload = "Represent this sentence for searching relevant passages: " + question.strip()[:800]
    enc = EMB_TOK(payload, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        vec = EMB(**enc).last_hidden_state[0, 0].detach().cpu().numpy().astype(np.float32)
    CACHE[key] = vec / (np.linalg.norm(vec) + 1e-8)
    return CACHE[key]


def sentences_of(stored):
    out, seen = [], set()
    for contract in stored:
        write_vec = getattr(contract, "write_vec", None)
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                carried = dict(ev)
                carried["_w"] = write_vec
                out.append(carried)
    return out


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


def top4(question, evs, qvec):
    q_toks = tokens(question)
    scored = []
    for ev in evs:
        text = clip(ev.get("text") or "")
        if not text:
            continue
        if text not in CACHE:
            embed_many([text])
        overlap = (len(q_toks & tokens(text)) / len(q_toks)) if q_toks else 0.0
        cos = float(np.dot(qvec, CACHE[text]))
        base = 0.45 * overlap + 0.55 * max(cos, 0.0)
        write_vec = ev.get("_w")
        if write_vec is not None:
            base *= 0.55 + 0.45 * max(float(np.dot(qvec, write_vec)), 0.0)
        scored.append((base, ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [ev for _, ev in scored[:4]]


def commit(question, evs):
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
    return best[1]


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            test_packets, pool_packets, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported"):
            for method in ("crag_plus", "trustmem"):
                own_lists = [sentences_of(stored) for stored in test_packets[setting][method]]
                pool = [sentences_of(stored) for stored in pool_packets["clean"][method]]
                texts = []
                for group in own_lists + pool:
                    texts.extend(clip(ev.get("text") or "") for ev in group)
                embed_many(texts)
                qvecs = [embed_query(item["question"]) for item in meta]
                immediate, h100, h1000 = [], [], []
                for index, item in enumerate(meta):
                    question = item["question"]
                    gold = item["answer"]
                    qvec = qvecs[index]
                    base = top4(question, own_lists[index], qvec)
                    immediate.append(token_f1(commit(question, base), gold))
                    start = index % max(len(pool), 1)
                    ordered = pool[start:] + pool[:start]
                    mem = list(own_lists[index])
                    for step, extra in enumerate(ordered, start=1):
                        mem.extend(extra)
                        if step == 100:
                            h100.append(token_f1(commit(question, top4(question, mem, qvec)), gold))
                        elif step == 1000:
                            h1000.append(token_f1(commit(question, top4(question, mem, qvec)), gold))
                            break
                print(
                    f"  {setting:12s} {method:10s}"
                    f" F1 {np.mean(immediate):.3f}"
                    f" H100 {np.mean(h100):.3f}"
                    f" H1000 {np.mean(h1000) if h1000 else float('nan'):.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
