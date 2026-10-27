import json
import numpy as np

d = json.load(open(r"results/scores_2wiki.json", encoding="utf-8"))
print("keys", len(d))


def boot(left, right, seed=3152, n_boot=2000):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


for setting in ("mixed", "clean", "unsupported"):
    left_key = f"{setting}:trustmem"
    if left_key not in d:
        continue
    tm = d[left_key]
    for other in ("crag", "crag_plus", "marag"):
        base = d[f"{setting}:{other}"]
        for name, key in (("f1", None), ("H1000", "1000")):
            if key is None:
                a, b = tm["f1s"], base["f1s"]
            else:
                a, b = tm["delayed_f1s"][key], base["delayed_f1s"][key]
            delta, lo, hi = boot(a, b)
            print(f"{setting:12s} {name:6s} vs {other:10s} {delta:+.3f} [{lo:+.3f}, {hi:+.3f}]")
