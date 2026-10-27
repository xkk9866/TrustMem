"""Answer from the single stored sentence the reader is most sure about."""

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
        for j in range(i, min(i + 12, len(seq_ids))):
            if seq_ids[j] != 1:
                break
            score = float(start[i] * end[j])
            if best is None or score > best[0]:
                best = (score, i, j)
    if best is None:
        return "", 0.0
    span = context[offsets[best[1]][0] : offsets[best[2]][1]].strip()
    return span, best[0]


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


def joint(question, evs):
    if not evs:
        return ""
    context = "\n".join(f"{ev.get('title', '')}: {ev.get('text', '')}" for ev in evs)
    pred, _ = solo(question, context)
    return pred


def one_sentence(question, evs):
    titles = {normalize_answer((ev.get("title") or "").split("(")[0]) for ev in evs}
    best = None
    for ev in evs:
        line = f"{ev.get('title', '')}: {ev.get('text', '')}"
        pred, conf = solo(question, line)
        needle = normalize_answer(pred)
        if not needle or any(needle == title or (len(needle) > 3 and needle in title) for title in titles if title):
            conf *= 0.25
        if best is None or conf > best[0]:
            best = (conf, pred)
    return "" if best is None else best[1]


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
        for setting in ("mixed", "unsupported", "incomplete", "clean"):
            scores = {name: [] for name in ("plus_joint", "plus_one", "mem_joint", "mem_one")}
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                plus = rank_top(question, rows(packets[setting]["crag_plus"][index]))
                mem = rank_top(question, rows(packets[setting]["trustmem"][index]))
                scores["plus_joint"].append(token_f1(joint(question, plus), gold))
                scores["plus_one"].append(token_f1(one_sentence(question, plus), gold))
                scores["mem_joint"].append(token_f1(joint(question, mem), gold))
                scores["mem_one"].append(token_f1(one_sentence(question, mem), gold))
            print(
                f"  {setting:12s}"
                f" plus {np.mean(scores['plus_joint']):.3f}/{np.mean(scores['plus_one']):.3f}"
                f"  mem {np.mean(scores['mem_joint']):.3f}/{np.mean(scores['mem_one']):.3f}",
                flush=True,
            )
            delta, lo, hi = bootstrap(scores["mem_one"], scores["plus_joint"])
            print(f"    mem_one-plus_joint {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
