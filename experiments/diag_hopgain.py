"""Where the citation hop changes the packet, and who wins."""
import json
import pickle
from pathlib import Path

root = Path("results")
with (root / "packets_cite60b.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
scores = json.loads((root / "scores_cite60b.json").read_text(encoding="utf-8"))


def texts(stored):
    out = []
    for contract in stored:
        for ev in contract.evidence or []:
            if ev.get("text"):
                out.append(ev["text"])
    return out


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    same = diff_win = diff_loss = diff_tie = 0
    win_sum = loss_sum = 0.0
    for index in range(len(meta)):
        tm = scores[f"{setting}:trustmem"]["f1s"][index]
        plus = scores[f"{setting}:crag_plus"]["f1s"][index]
        identical = texts(packets[setting]["trustmem"][index]) == texts(packets[setting]["crag_plus"][index])
        if identical:
            same += 1
            continue
        delta = tm - plus
        if delta > 0.05:
            diff_win += 1
            win_sum += delta
        elif delta < -0.05:
            diff_loss += 1
            loss_sum += delta
        else:
            diff_tie += 1
    print(
        f"{setting:12s} identical={same:2d} hop_win={diff_win:2d} ({win_sum:+.2f}) hop_loss={diff_loss:2d} ({loss_sum:+.2f}) hop_tie={diff_tie:2d}"
    )
