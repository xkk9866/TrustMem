"""Sentences on items where CRAG beats TrustMem."""
import json
import pickle
from pathlib import Path

from run_trustmem import gold_recall

root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_cite60.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
scores = json.loads((root / "results" / "scores_cite60.json").read_text(encoding="utf-8"))

setting = "unsupported"
gaps = []
for index in range(60):
    tm = scores[f"{setting}:trustmem"]["f1s"][index]
    crag = scores[f"{setting}:crag"]["f1s"][index]
    gaps.append((crag - tm, index, tm, crag))
gaps.sort(reverse=True)
for gap, index, tm, crag in gaps[:6]:
    if gap < 0.2:
        break
    print(f"\nGAP {gap:.2f} tm={tm:.2f} crag={crag:.2f}")
    print("Q:", meta[index]["question"])
    print("A:", meta[index]["answer"])
    for method in ("crag", "trustmem"):
        print(" ", method)
        seen = set()
        for contract in packets[setting][method]:
            pass
        for contract in packets[setting][method][index]:
            for ev in contract.evidence or []:
                text = ev.get("text", "")
                if not text or text in seen:
                    continue
                seen.add(text)
                flag = " *" if gold_recall(text, meta[index]["answer"]) > 0.5 else ""
                print(f"   {ev.get('title','')[:32]:32s}{flag} {text[:150]}")
