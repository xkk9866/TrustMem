"""Pilot citation-hop admission on the same 60 Hotpot questions as the previous pilot."""
import os

import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)
os.environ.setdefault("DURABLE_DEVICE", "cuda:1")

from run_durable import base_cfg, evaluate_block, load_slices
from run_trustmem import load_models

models = load_models("cuda:1")
_, test, pool = load_slices("hotpot", 120, 4)
print("lens", len(test), len(pool), flush=True)
evaluate_block(
    models,
    test,
    pool,
    ["crag", "crag_plus", "trustmem"],
    ["mixed", "unsupported", "incomplete", "conflict"],
    base_cfg(),
    n_agents=2,
    mode="random",
    seed=0,
    horizons=(4,),
    label="cite120",
)
print("DONE", flush=True)
