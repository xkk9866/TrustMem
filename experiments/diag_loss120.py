"""Loss items on the 120-question mixed run."""
import json
import pickle
from pathlib import Path

from run_trustmem import gold_recall, tokens

root = Path("results")
with (root / "packets_cite120.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
scores = json.loads((root / "scores_cite120.json").read_text(encoding="utf-8"))


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                out.append((ev.get("title") or "", text))
    return out


setting = "mixed"
for index in range(len(meta)):
    tm = scores[f"{setting}:trustmem"]["f1s"][index]
    plus = scores[f"{setting}:crag_plus"]["f1s"][index]
    if tm - plus < 0.2:
        continue
    kind = "WIN"
    q = tokens(meta[index]["question"])
    print(f"\n{kind} {tm - plus:+.2f}")
    print("Q:", meta[index]["question"])
    print("A:", meta[index]["answer"])
    base = rows(packets[setting]["crag_plus"][index])
    hop = rows(packets[setting]["trustmem"][index])
    base_set = {text for _, text in base}
    hop_set = {text for _, text in hop}
    for title, text in hop:
        if text not in base_set:
            hit = "Q" if tokens(title.split("(")[0]) & q else " "
            flag = "*" if gold_recall(text, meta[index]["answer"]) > 0.5 else " "
            print(f"  ADD {hit}{flag} {title[:36]} | {text[:120]}")
    for title, text in base:
        if text not in hop_set:
            hit = "Q" if tokens(title.split("(")[0]) & q else " "
            flag = "*" if gold_recall(text, meta[index]["answer"]) > 0.5 else " "
            print(f"  DROP{hit}{flag} {title[:36]} | {text[:120]}")
