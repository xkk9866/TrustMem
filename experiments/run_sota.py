"""Full TrustMem readout on the stored packets.

Same frozen span reader for every column. TrustMem evaluates a comparison or a
follow when one resolves, otherwise reads the pages cited by the title named in
the question. The packet columns read one sentence at a time.
"""

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from diag_compare2 import choices_of, pick
from diag_holdrank import overlap, packet_rows
from run_durable import load_slices
from run_trustmem import normalize_answer, token_f1
from witness_memory import cited_pages, express

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("TRUSTMEM_DEVICE", "cuda:1")
SETTINGS = ("mixed", "unsupported", "incomplete", "clean", "conflict")
PACKETS = ("marag", "crag", "crag_plus")


def load_reader(device: str):
    index = int(device.split(":")[1]) if device.startswith("cuda") else 0
    if device.startswith("cuda"):
        fraction = float(os.environ.get("TRUSTMEM_FRACTION", "0.12"))
        torch.cuda.set_per_process_memory_fraction(fraction, index)
    name = "deepset/deberta-v3-base-squad2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForQuestionAnswering.from_pretrained(name).to(device).eval()
    cache = {}

    def answer(question: str, context: str):
        if not context.strip():
            return "", 0.0
        key = (question, context)
        hit = cache.get(key)
        if hit is not None:
            return hit
        enc = tok(
            question,
            context,
            truncation="only_second",
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        seq_ids = enc.sequence_ids(0)
        batch = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**batch)
        start = torch.softmax(out.start_logits[0], dim=-1).detach().cpu().numpy()
        end = torch.softmax(out.end_logits[0], dim=-1).detach().cpu().numpy()
        best = None
        limit = len(seq_ids)
        for i in range(limit):
            if seq_ids[i] != 1:
                continue
            for j in range(i, min(i + 24, limit)):
                if seq_ids[j] != 1:
                    break
                score = float(start[i] * end[j])
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best is None:
            cache[key] = ("", 0.0)
            return cache[key]
        char_s, char_e = offsets[best[1]][0], offsets[best[2]][1]
        if char_s is None or char_e is None or char_e <= char_s:
            cache[key] = ("", 0.0)
            return cache[key]
        cache[key] = (context[char_s:char_e].strip(), best[0])
        return cache[key]

    def score_span(question: str, context: str, span: str):
        """Probability of a specific span under the same frozen reader."""
        if not context.strip() or not span or span.lower() not in context.lower():
            return 0.0
        enc = tok(
            question,
            context,
            truncation="only_second",
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        seq_ids = enc.sequence_ids(0)
        batch = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**batch)
        start = torch.softmax(out.start_logits[0], dim=-1).detach().cpu().numpy()
        end = torch.softmax(out.end_logits[0], dim=-1).detach().cpu().numpy()
        begin = context.lower().find(span.lower())
        finish = begin + len(span)
        start_at, end_at = None, None
        for index, (char_s, char_e) in enumerate(offsets):
            if seq_ids[index] != 1 or char_s is None or char_e is None:
                continue
            if start_at is None and char_s <= begin < char_e:
                start_at = index
            if char_s < finish <= char_e or (char_s <= finish - 1 < char_e):
                end_at = index
        if start_at is None or end_at is None or end_at < start_at or end_at >= start_at + 24:
            return 0.0
        return float(start[start_at] * end[end_at])

    answer.score_span = score_span
    return answer


def packet_evidence(stored):
    rows, seen = [], set()
    for contract in stored:
        for ev in contract.evidence or []:
            key = (ev.get("title") or "", ev.get("text") or "")
            if key[1] and key not in seen:
                seen.add(key)
                rows.append(key)
    return rows


def page_rows(pages, n=4):
    rows = []
    for doc in pages:
        if not doc:
            continue
        for sent in doc["sentences"][:n]:
            if sent:
                rows.append((doc["title"], sent))
    return rows


def commit_scored(answer, question, rows, penalty=True):
    best = (0.0, "")
    titles = {normalize_answer(title.split("(")[0]) for title, _ in rows}
    for title, text in rows:
        pred, conf = answer(question, f"{title}: {text}")
        needle = normalize_answer(pred)
        if penalty and needle and any(
            needle == name or (len(needle) > 3 and needle in name) for name in titles if name
        ):
            conf *= 0.25
        if conf > best[0]:
            best = (conf, pred)
    return best


def commit(answer, question, rows, penalty=True):
    return commit_scored(answer, question, rows, penalty)[1]


def read_page(answer, question, doc):
    if not doc or not doc["sentences"]:
        return ""
    context = " ".join(f"{doc['title']}: {sent}" for sent in doc["sentences"][:6])
    pred, _ = answer(question, context)
    return pred


def bootstrap(left, right):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def mean(scores):
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(answer, dataset):
    _, test, _ = load_slices(dataset, 120, 4)
    by_q = {item["question"]: item for item in test}
    with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, pool, meta = pickle.load(handle)
    present = [name for name in SETTINGS if name in packets]
    report = {"n": len(meta), "settings": {}}
    print(f"\n== {dataset} n={len(meta)} settings={present}", flush=True)
    for setting in present:
        columns = {
            name: []
            for name in ("marag", "crag", "faithful", "algebra", "cited", "gated", "union", "best", "pages")
        }
        fired = {"compare": 0, "follow": 0, "bridge": 0, "cited": 0, "packet": 0, "best": 0}
        for index, item in enumerate(meta):
            question, gold = item["question"], item["answer"]
            docs = by_q[question]["docs"]
            faithful_rows = packet_evidence(packets[setting]["crag_plus"][index])
            faithful_conf, faithful_span = commit_scored(answer, question, faithful_rows)
            heads = {
                "marag": commit(answer, question, packet_evidence(packets[setting]["marag"][index])),
                "crag": commit(answer, question, packet_evidence(packets[setting]["crag"][index])),
                "faithful": faithful_span,
            }
            value, how, opened = express(question, docs, lambda q, doc: read_page(answer, q, doc))
            links = [] if value else cited_pages(question, docs)
            link_rows = page_rows(links)
            if value:
                algebra = cited = gated = union = best = value
                used = opened
                kind = how
            else:
                algebra = heads["faithful"]
                if links:
                    link_conf, link_span = commit_scored(answer, question, link_rows)
                    cited = link_span
                    anchor_title = links[0]["title"].lower()
                    packet_titles = {title.lower() for title, _ in faithful_rows}
                    gated = link_span if anchor_title not in packet_titles else heads["faithful"]
                    union = commit(answer, question, faithful_rows + link_rows)
                    if link_conf > faithful_conf:
                        best = link_span
                        kind = "best"
                    else:
                        best = heads["faithful"]
                        kind = "packet"
                    used = links
                else:
                    cited = gated = union = best = heads["faithful"]
                    used = []
                    kind = "packet"
            if value and how == "compare":
                pages = commit(answer, question, page_rows(opened)) or heads["faithful"]
            elif (value and how in {"follow", "bridge"}) or links:
                pages = commit(answer, question, page_rows(used)) or heads["faithful"]
            else:
                pages = heads["faithful"]
            fired[kind if kind in fired else "packet"] += 1
            columns["marag"].append(token_f1(heads["marag"], gold))
            columns["crag"].append(token_f1(heads["crag"], gold))
            columns["faithful"].append(token_f1(heads["faithful"], gold))
            columns["algebra"].append(token_f1(algebra, gold))
            columns["cited"].append(token_f1(cited, gold))
            columns["gated"].append(token_f1(gated, gold))
            columns["union"].append(token_f1(union, gold))
            columns["best"].append(token_f1(best, gold))
            columns["pages"].append(token_f1(pages, gold))
        block = {name: mean(columns[name]) for name in columns}
        block["fired"] = fired
        pairs = (
            ("best", "faithful"),
            ("best", "algebra"),
            ("best", "pages"),
            ("gated", "faithful"),
            ("union", "faithful"),
            ("cited", "faithful"),
            ("algebra", "faithful"),
            ("best", "crag"),
            ("best", "marag"),
        )
        block["delta"] = {}
        for left, right in pairs:
            delta, lo, hi = bootstrap(columns[left], columns[right])
            block["delta"][f"{left}-{right}"] = {"mean": delta, "lo": lo, "hi": hi}
        report["settings"][setting] = block
        print(
            f"  {setting:12s} marag {block['marag']:.3f} crag {block['crag']:.3f} "
            f"faithful {block['faithful']:.3f} algebra {block['algebra']:.3f} "
            f"cited {block['cited']:.3f} gated {block['gated']:.3f} "
            f"union {block['union']:.3f} best {block['best']:.3f} pages {block['pages']:.3f} {fired}",
            flush=True,
        )
        for key, delta in block["delta"].items():
            mark = " *" if delta["lo"] > 0 else ""
            print(f"    {key} {delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]{mark}", flush=True)
    if dataset == "2wiki" and pool and "clean" in pool:
        report["horizon"] = horizon(answer, by_q, packets, pool, meta)
    return report


def horizon(answer, by_q, packets, pool, meta):
    later = []
    for stored in pool["clean"]["crag_plus"]:
        later.extend(packet_rows(stored))
    print(f"  horizon later {len(later)}", flush=True)
    out = {"later": len(later), "settings": {}}
    for setting in ("mixed", "unsupported", "incomplete", "clean"):
        if setting not in packets:
            continue
        base, live, first = [], [], 0
        for index, item in enumerate(meta):
            question, gold = item["question"], item["answer"]
            rows = packet_evidence(packets[setting]["crag_plus"][index])
            head = commit(answer, question, rows, penalty=False)
            base.append(token_f1(head, gold))
            stored = pick(question, by_q[question]["docs"])
            opts = choices_of(question)
            ranked = [(overlap(question, f"{title} {text}"), 0, "") for title, text in rows + later]
            if stored and len(opts) == 2:
                text = f"Between {opts[0]} and {opts[1]}, the comparison selects {stored}."
                ranked.append((overlap(question, text), 1, stored))
            ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
            if ranked and ranked[0][2]:
                first += 1
                pred = ranked[0][2]
            else:
                pred = head
            live.append(token_f1(pred, gold))
        delta, lo, hi = bootstrap(live, base)
        out["settings"][setting] = {
            "commitment": mean(base),
            "stored": mean(live),
            "first": first,
            "delta": {"mean": delta, "lo": lo, "hi": hi},
        }
        mark = " *" if lo > 0 else ""
        print(
            f"  horizon {setting:12s} stored {mean(live):.3f} commitment {mean(base):.3f} "
            f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark} first {first}",
            flush=True,
        )
    return out


def main():
    print("device", DEVICE, flush=True)
    answer = load_reader(DEVICE)
    report = {"device": DEVICE, "reader": "deepset/deberta-v3-base-squad2", "datasets": {}}
    for dataset in ("2wiki", "hotpot"):
        report["datasets"][dataset] = evaluate(answer, dataset)
        path = ROOT / "results" / "sota_read.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
