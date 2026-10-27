"""Attested wide log with verbatim provenance only.

A quote that is not on the cited page is replaced by a read of that page.
Near-copy quotes are not kept by an entailment check. Same reader, same log
budget, same baselines.
"""

import os

import torch

torch.cuda.set_per_process_memory_fraction(0.18, 0)
os.environ["DURABLE_DEVICE"] = "cuda:0"
os.environ["TRUSTMEM_PACKET"] = "wide"
os.environ["TRUSTMEM_SEMANTIC"] = "0"
os.environ["TRUSTMEM_BIND"] = "0"

from run_durable import base_cfg, evaluate_block, load_slices
from run_trustmem import load_models

models = load_models("cuda:0")
cfg = base_cfg()
cfg["semantic"] = False
print("semantic", cfg["semantic"], flush=True)
for name in ("hotpot", "2wiki"):
    _, test, pool = load_slices(name, 120, 4)
    print(name, len(test), len(pool), flush=True)
    evaluate_block(
        models,
        test,
        pool,
        ["crag_plus", "trustmem"],
        ["mixed", "unsupported", "incomplete", "clean"],
        cfg,
        n_agents=2,
        mode="random",
        seed=0,
        horizons=(4,),
        label=f"sem0-{name}",
    )
print("DONE", flush=True)
