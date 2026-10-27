"""Test-slice readout for inside-out witness evaluation.

The reported method is gated: an order is kept, and a pointer is kept only
when the residual read is strictly more confident than the page-faithful
packet. The ungated pointer column is the same derivation without that check.
"""

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from derivation import derive
from run_durable import load_slices
from run_sota import (
    DEVICE,
    ROOT,
    SETTINGS,
    bootstrap,
    commit,
    commit_scored,
    load_reader,
    mean,
    packet_evidence,
    page_rows,
)
from run_trustmem import token_f1
OUT = ROOT / "results" / "chain_read.json"


def read_scored(answer, question, doc):
    if not doc:
        return "", 0.0
    return commit_scored(answer, question, page_rows([doc], n=6))[::-1]


def evaluate(answer, dataset):
    _, test, _ = load_slices(dataset, 120, 4)
    by_q = {item["question"]: item for item in test}
    with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    present = [name for name in SETTINGS if name in packets]
    report = {"n": len(meta), "settings": {}}
    print(f"\n== {dataset} n={len(meta)}", flush=True)

    def read(question, doc):
        span, conf = read_scored(answer, question, doc)
        return span, conf

    for setting in present:
        columns = {name: [] for name in ("marag", "crag", "faithful", "chain", "gated")}
        fired = {"order": 0, "kept": 0, "rejected": 0, "packet": 0}
        for index, item in enumerate(meta):
            question, gold = item["question"], item["answer"]
            docs = by_q[question]["docs"]
            faithful_rows = packet_evidence(packets[setting]["crag_plus"][index])
            faithful_conf, faithful_span = commit_scored(answer, question, faithful_rows)
            value, kind, _, conf = derive(question, docs, read)
            if kind == "order":
                chain = gated = value
                fired["order"] += 1
            elif kind == "pointer" and conf > faithful_conf:
                chain = gated = value
                fired["kept"] += 1
            elif kind == "pointer":
                chain = value
                gated = faithful_span
                fired["rejected"] += 1
            else:
                chain = gated = faithful_span
                fired["packet"] += 1
            columns["marag"].append(
                token_f1(commit(answer, question, packet_evidence(packets[setting]["marag"][index])), gold)
            )
            columns["crag"].append(
                token_f1(commit(answer, question, packet_evidence(packets[setting]["crag"][index])), gold)
            )
            columns["faithful"].append(token_f1(faithful_span, gold))
            columns["chain"].append(token_f1(chain, gold))
            columns["gated"].append(token_f1(gated, gold))
        block = {name: mean(columns[name]) for name in columns}
        block["fired"] = fired
        block["delta"] = {}
        for left, right in (("gated", "faithful"), ("gated", "crag"), ("gated", "marag"), ("chain", "faithful"), ("chain", "gated")):
            delta, lo, hi = bootstrap(columns[left], columns[right])
            block["delta"][f"{left}-{right}"] = {"mean": delta, "lo": lo, "hi": hi}
        report["settings"][setting] = block
        print(
            f"  {setting:12s} marag {block['marag']:.3f} crag {block['crag']:.3f} "
            f"faithful {block['faithful']:.3f} chain {block['chain']:.3f} gated {block['gated']:.3f} {fired}",
            flush=True,
        )
        for key, delta in block["delta"].items():
            mark = " *" if delta["lo"] > 0 else ""
            print(f"    {key} {delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]{mark}", flush=True)
    return report


def main():
    print("device", DEVICE, flush=True)
    answer = load_reader(DEVICE)
    report = {
        "device": DEVICE,
        "reader": "deepset/deberta-v3-base-squad2",
        "method": "gated",
        "datasets": {},
    }
    for dataset in ("2wiki", "hotpot"):
        report["datasets"][dataset] = evaluate(answer, dataset)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
