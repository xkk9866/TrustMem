"""Elect an answer from per-sentence reads of the attested log.

Each sentence proposes a span. Compatible spans accumulate reader confidence.
A second rule drops a span that only repeats a page title. If every proposal
is below 0.05, the paper's four-sentence read is used.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.20, 1)

from run_trustmem import Contract, normalize_answer, reader_from_contracts, token_f1, tokens

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:1"


class Bundle:
    pass


def load():
    bundle = Bundle()
    bundle.device = DEVICE
    bundle.qa_tok = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-squad2")
    bundle.qa = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-base-squad2").to(DEVICE).eval()
    bundle.emb_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    bundle.emb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEVICE).eval()
    bundle.emb_cache = {}
    bundle.nli_cache = {}
    return bundle


MODELS = load()


def embed(text, query):
    raw = text.strip()[:800]
    key = (query, raw)
    if key in MODELS.emb_cache:
        return MODELS.emb_cache[key]
    payload = raw
    if query:
        payload = "Represent this sentence for searching relevant passages: " + raw
    enc = MODELS.emb_tok(payload, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        vec = MODELS.emb(**enc).last_hidden_state[0, 0].detach().cpu().numpy().astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    MODELS.emb_cache[key] = vec
    return vec


def solo(question, context):
    enc = MODELS.qa_tok(
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
        out = MODELS.qa(**batch)
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


def rank_top(question, evs, k=4):
    if not evs:
        return []
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
    packed = Contract("", evs, "readout", "read", 0.5, 0, 10**9, "none", False)
    # reader_from_contracts expects models.answer. Bind a thin wrapper.
    pred, _, _ = reader_from_contracts(MODELS, question, [packed])
    return pred


def answer_api(question, context):
    return solo(question, context)


MODELS.answer = answer_api


def compatible(left, right):
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= 4 and left in right:
        return True
    if len(right) >= 4 and right in left:
        return True
    return False


def elect(question, evs, drop_titles: bool):
    titles = {normalize_answer((ev.get("title") or "").split("(")[0]) for ev in evs}
    titles.discard("")
    clusters = []
    for ev in evs:
        line = f"{ev.get('title', '')}: {ev.get('text', '')}".strip()
        pred, conf = solo(question, line)
        needle = normalize_answer(pred)
        if not needle or conf < 0.01:
            continue
        if drop_titles and any(needle == title or (len(needle) >= 4 and needle in title) for title in titles):
            continue
        placed = False
        for cluster in clusters:
            if compatible(needle, cluster["needle"]):
                cluster["score"] += conf
                if conf > cluster["conf"]:
                    cluster["conf"] = conf
                    cluster["pred"] = pred
                    cluster["needle"] = needle if len(needle) <= len(cluster["needle"]) else cluster["needle"]
                placed = True
                break
        if not placed:
            clusters.append({"needle": needle, "pred": pred, "score": conf, "conf": conf})
    if not clusters or max(item["score"] for item in clusters) < 0.05:
        return None
    return max(clusters, key=lambda item: item["score"])["pred"]


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
            scores = {name: [] for name in ("crag_plus", "trustmem", "sum", "blind")}
            for index, item in enumerate(meta):
                question = item["question"]
                gold = item["answer"]
                plus = rank_top(question, rows(packets[setting]["crag_plus"][index]))
                mem_ev = rows(packets[setting]["trustmem"][index])
                mem = rank_top(question, mem_ev)
                pred_sum = elect(question, mem_ev, False) or joint(question, mem)
                pred_blind = elect(question, mem_ev, True) or joint(question, mem)
                scores["crag_plus"].append(token_f1(joint(question, plus), gold))
                scores["trustmem"].append(token_f1(joint(question, mem), gold))
                scores["sum"].append(token_f1(pred_sum, gold))
                scores["blind"].append(token_f1(pred_blind, gold))
            print(
                f"  {setting:12s} plus {np.mean(scores['crag_plus']):.3f}"
                f"  mem {np.mean(scores['trustmem']):.3f}"
                f"  sum {np.mean(scores['sum']):.3f}"
                f"  blind {np.mean(scores['blind']):.3f}",
                flush=True,
            )
            for name in ("sum", "blind"):
                for base in ("crag_plus", "trustmem"):
                    delta, lo, hi = bootstrap(scores[name], scores[base])
                    print(f"    {name}-{base}: {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
