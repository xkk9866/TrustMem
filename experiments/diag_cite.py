"""Did citation hop shrink the packet without dropping the answer?"""
import pickle
from pathlib import Path

from run_trustmem import gold_recall

root = Path(__file__).resolve().parents[1]


def load(name):
    with (root / "results" / name).open("rb") as handle:
        test_packets, _, meta = pickle.load(handle)
    return test_packets, meta


def summarize(packets, meta, setting, method):
    n = gold = 0.0
    count = 0
    for index, stored in enumerate(packets[setting][method]):
        texts = []
        seen = set()
        for contract in stored:
            for ev in contract.evidence or []:
                text = ev.get("text", "")
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
        count += 1
        n += len(texts)
        gold += gold_recall(" ".join(texts), meta[index]["answer"])
    return n / count, gold / count


new_packets, new_meta = load("packets_cite60.pkl")
old_packets, old_meta = load("packets_pilot.pkl")
for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    print(setting)
    for label, packets, meta in (("old", old_packets, old_meta), ("new", new_packets, new_meta)):
        for method in ("crag_plus", "trustmem"):
            if method not in packets[setting]:
                continue
            n, gold = summarize(packets, meta, setting, method)
            print(f"  {label:3s} {method:10s} n={n:.1f} gold={gold:.3f}")
