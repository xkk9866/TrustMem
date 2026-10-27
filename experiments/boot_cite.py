"""Paired bootstrap on the 60-question citation-hop pilot."""
import json
from pathlib import Path

import numpy as np

from run_durable import bootstrap_delta

scores = json.loads(Path("results/scores_cite120.json").read_text(encoding="utf-8"))
for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    tm = scores[f"{setting}:trustmem"]["f1s"]
    for other in ("crag", "crag_plus"):
        stat = bootstrap_delta(tm, scores[f"{setting}:{other}"]["f1s"])
        print(f"{setting:12s} vs {other:10s} {stat['delta']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}]")
