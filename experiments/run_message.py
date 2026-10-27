"""Test readout. Specialists read only their own pages.

The reported method keeps an order, and keeps a binding only when its
confidence exceeds the corrected packet. Carry forwards the original question
across the same addresses. Open reading asks that question on every page.
"""
import json
import os
import pickle

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRUSTMEM_DEVICE", "cuda:1")
os.environ.setdefault("TRUSTMEM_FRACTION", "0.08")

from message import carry, deliver
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
from run_trustmem import normalize_answer, token_f1

OUT = ROOT / "results" / "message_read.json"


def read_scored(answer, question, doc):
    if not doc:
        return "", 0.0
    return commit_scored(answer, question, page_rows([doc], n=6))[::-1]


def open_read(answer, question, docs):
    best = (0.0, "")
    for doc in docs:
        conf, span = commit_scored(answer, question, page_rows([doc], n=6))
        if conf > best[0]:
            best = (conf, span)
    return best


def evaluate(answer, dataset):
    _, test, _ = load_slices(dataset, 120, 4)
    by_q = {item["question"]: item for item in test}
    with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    present = [name for name in SETTINGS if name in packets]
    report = {"n": len(meta), "settings": {}}
    print(f"\n== {dataset} n={len(meta)}", flush=True)

    def read(question, doc):
        return read_scored(answer, question, doc)

    read.score_span = answer.score_span
    derived = []
    carried = []
    opened = []
    order_exact = order_fired = 0
    for item in meta:
        question = item["question"]
        source = by_q[question]
        docs = source["docs"]
        qid = source["id"]
        derived.append(deliver(question, docs, read, qid))
        carried.append(carry(question, docs, read, qid))
        opened.append(open_read(answer, question, docs))
        value, kind, _, _ = derived[-1]
        if kind == "order":
            order_fired += 1
            if normalize_answer(value) == normalize_answer(item["answer"]):
                order_exact += 1
        if len(derived) % 20 == 0:
            print(f"  prepared {len(derived)}", flush=True)
    print(f"  orders {order_exact} exact of {order_fired} fired", flush=True)
    report["orders"] = {"fired": order_fired, "exact": order_exact}

    names = ("marag", "crag", "faithful", "open", "carry", "chain", "gated")
    for setting in present:
        columns = {name: [] for name in names}
        fired = {"order": 0, "kept": 0, "rejected": 0, "packet": 0}
        for index, item in enumerate(meta):
            question, gold = item["question"], item["answer"]
            faithful_rows = packet_evidence(packets[setting]["crag_plus"][index])
            faithful_conf, faithful_span = commit_scored(answer, question, faithful_rows)
            value, kind, _, conf = derived[index]
            if kind == "order":
                chain = gated = value
                fired["order"] += 1
            elif kind == "denote" and conf > faithful_conf:
                chain = gated = value
                fired["kept"] += 1
            elif kind == "denote":
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
            columns["open"].append(token_f1(opened[index][1], gold))
            columns["carry"].append(token_f1(carried[index][0], gold))
            columns["chain"].append(token_f1(chain, gold))
            columns["gated"].append(token_f1(gated, gold))
        block = {name: mean(columns[name]) for name in names}
        block["fired"] = fired
        block["delta"] = {}
        pairs = (
            ("gated", "faithful"),
            ("gated", "open"),
            ("gated", "carry"),
            ("gated", "crag"),
            ("gated", "marag"),
            ("chain", "gated"),
            ("carry", "open"),
            ("open", "faithful"),
        )
        for left, right in pairs:
            delta, lo, hi = bootstrap(columns[left], columns[right])
            block["delta"][f"{left}-{right}"] = {"mean": delta, "lo": lo, "hi": hi}
        report["settings"][setting] = block
        print(
            f"  {setting:12s} faithful {block['faithful']:.3f} open {block['open']:.3f} "
            f"carry {block['carry']:.3f} chain {block['chain']:.3f} gated {block['gated']:.3f} {fired}",
            flush=True,
        )
        for key, delta in block["delta"].items():
            mark = " *" if delta["lo"] > 0 else ""
            print(f"    {key} {delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]{mark}", flush=True)
    return report


def second_seed(answer, dataset, seed=2027):
    """Another 120-question slice, with no planted faults. The slice seed is not 3152."""
    import random

    from datasets import load_dataset
    from run_trustmem import answer_supported, parse_example

    if dataset == "hotpot":
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
    else:
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    parsed = []
    for ex in raw:
        item = parse_example(ex)
        if dataset == "hotpot" and item["type"] != "bridge":
            continue
        if answer_supported(item):
            parsed.append(item)
    rng = random.Random(seed)
    rng.shuffle(parsed)
    test = parsed[36:156]
    print(f"\n== {dataset} seed {seed} n={len(test)}", flush=True)

    def read(question, doc):
        return read_scored(answer, question, doc)

    read.score_span = answer.score_span
    columns = {name: [] for name in ("message", "carry", "open")}
    for index, item in enumerate(test):
        question, gold, docs, qid = item["question"], item["answer"], item["docs"], item["id"]
        value, _kind, _, _conf = deliver(question, docs, read, qid)
        carried, _ = carry(question, docs, read, qid)
        _conf, opened = open_read(answer, question, docs)
        columns["message"].append(token_f1(value, gold))
        columns["carry"].append(token_f1(carried, gold))
        columns["open"].append(token_f1(opened, gold))
        if (index + 1) % 20 == 0:
            print(f"  seed prepared {index + 1}", flush=True)
    block = {name: mean(columns[name]) for name in columns}
    block["delta"] = {}
    for left, right in (("message", "open"), ("message", "carry"), ("carry", "open")):
        delta, lo, hi = bootstrap(columns[left], columns[right])
        block["delta"][f"{left}-{right}"] = {"mean": delta, "lo": lo, "hi": hi}
    print(
        f"  seed {seed} message {block['message']:.3f} carry {block['carry']:.3f} open {block['open']:.3f}",
        flush=True,
    )
    for key, delta in block["delta"].items():
        mark = " *" if delta["lo"] > 0 else ""
        print(f"    {key} {delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]{mark}", flush=True)
    return block


def main():
    print("device", DEVICE, "fraction", os.environ.get("TRUSTMEM_FRACTION"), flush=True)
    answer = load_reader(DEVICE)
    report = {
        "device": DEVICE,
        "reader": "deepset/deberta-v3-base-squad2",
        "method": "message binding, owner-only reads",
        "datasets": {},
    }
    for dataset in ("2wiki", "hotpot"):
        report["datasets"][dataset] = evaluate(answer, dataset)
        if os.environ.get("TRUSTMEM_SKIP_SEED") != "1":
            report["datasets"][dataset]["seed2027"] = second_seed(answer, dataset)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
