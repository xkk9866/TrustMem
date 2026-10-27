"""Print the questions where the hop changed F1."""
import json
import pickle
from pathlib import Path

from run_trustmem import gold_recall

root = Path("results")
with (root / "packets_cite60b.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
scores = json.loads((root / "scores_cite60b.json").read_text(encoding="utf-8"))


def texts(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            title = ev.get("title", "")
            if text and text not in seen:
                seen.add(text)
                out.append((title, text))
    return out


setting = "mixed"
for index in range(len(meta)):
    tm = scores[f"{setting}:trustmem"]["f1s"][index]
    plus = scores[f"{setting}:crag_plus"]["f1s"][index]
    if abs(tm - plus) < 0.05:
        continue
    print(f"\nDELTA {tm - plus:+.2f} tm={tm:.2f} plus={plus:.2f}")
    print("Q:", meta[index]["question"])
    print("A:", meta[index]["answer"])
    tm_set = {text for _, text in texts(packets[setting]["trustmem"][index])}
    plus_rows = texts(packets[setting]["crag_plus"][index])
    tm_rows = texts(packets[setting]["trustmem"][index])
    print(" only trustmem:")
    for title, text in tm_rows:
        if text not in {t for _, t in plus_rows}:
            flag = " *" if gold_recall(text, meta[index]["answer"]) > 0.5 else ""
            print(f"   {title[:40]}{flag}: {text[:160]}")
    print(" dropped from crag+:")
    for title, text in plus_rows:
        if text not in tm_set:
            flag = " *" if gold_recall(text, meta[index]["answer"]) > 0.5 else ""
            print(f"   {title[:40]}{flag}: {text[:160]}")
