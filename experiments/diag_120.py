"""How much F1 the hop gives and takes on the 120-question run."""
import json
import pickle
from pathlib import Path

root = Path("results")
with (root / "packets_cite120.pkl").open("rb") as handle:
    packets, _, meta = pickle.load(handle)
scores = json.loads((root / "scores_cite120.json").read_text(encoding="utf-8"))


def texts(stored):
    return [ev.get("text", "") for c in stored for ev in (c.evidence or []) if ev.get("text")]


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    same = win = loss = 0
    win_sum = loss_sum = 0.0
    for index in range(len(meta)):
        tm = scores[f"{setting}:trustmem"]["f1s"][index]
        plus = scores[f"{setting}:crag_plus"]["f1s"][index]
        if texts(packets[setting]["trustmem"][index]) == texts(packets[setting]["crag_plus"][index]):
            same += 1
            continue
        delta = tm - plus
        if delta > 0.05:
            win += 1
            win_sum += delta
        elif delta < -0.05:
            loss += 1
            loss_sum += delta
    print(
        f"{setting:12s} same={same:3d} win={win:2d} {win_sum:+.2f} loss={loss:2d} {loss_sum:+.2f} net={win_sum+loss_sum:+.2f}"
    )
