"""Full Hotpot and 2Wiki run for citation-hop TrustMem. Does not overwrite durable_results.json."""
import json
import os

import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)
os.environ["DURABLE_DEVICE"] = "cuda:1"

from run_durable import HORIZONS, base_cfg, bootstrap_delta, evaluate_block, load_slices
from run_trustmem import load_models

ROOT_OUT = os.path.join(os.path.dirname(__file__), "..", "results", "cite_results.json")

models = load_models("cuda:1")
cfg = base_cfg()
methods = ["marag", "crag", "crag_plus", "trustmem"]
settings = ["clean", "mixed", "unsupported", "incomplete", "stale", "conflict", "paraphrase"]
output = {
    "config": {key: cfg[key] for key in ("theta_entail", "theta_app", "semantic_tau", "qscore")},
    "horizons": list(HORIZONS),
    "datasets": {},
}
details = {}
for name, n_pool in (("hotpot", 1000), ("2wiki", 1000)):
    _, test, pool = load_slices(name, 120, n_pool)
    print(name, "test", len(test), "pool", len(pool), flush=True)
    summary, detail = evaluate_block(
        models,
        test,
        pool,
        methods,
        settings,
        cfg,
        n_agents=2,
        mode="random",
        seed=0,
        horizons=HORIZONS,
        label=f"cite-{name}",
    )
    output["datasets"][name] = summary
    details[name] = detail
    with open(ROOT_OUT, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print("wrote", name, flush=True)

output["bootstrap"] = {}
for dataset in ("hotpot", "2wiki"):
    block = details[dataset]
    for setting in ("mixed", "unsupported", "incomplete", "conflict"):
        left = block[setting]["trustmem"]
        for other in ("crag", "crag_plus", "marag"):
            right = block[setting][other]
            key = f"{dataset}:{setting}:f1_vs_{other}"
            output["bootstrap"][key] = bootstrap_delta(left["f1s"], right["f1s"])
            if "1000" in left["delayed_f1s"]:
                output["bootstrap"][f"{dataset}:{setting}:H1000_vs_{other}"] = bootstrap_delta(
                    left["delayed_f1s"]["1000"], right["delayed_f1s"]["1000"]
                )
print("bootstrap", json.dumps(output["bootstrap"]), flush=True)
with open(ROOT_OUT, "w", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2)
print("DONE", flush=True)
