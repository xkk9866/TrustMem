"""Wide log versus the citation packet, on the first 60 questions of each dataset."""
import os

import torch

torch.cuda.set_per_process_memory_fraction(0.30, 1)
os.environ["DURABLE_DEVICE"] = "cuda:1"
os.environ["TRUSTMEM_PACKET"] = "wide"

from run_durable import base_cfg, evaluate_block, load_slices
from run_trustmem import load_models

models = load_models("cuda:1")
cfg = base_cfg()
cfg["semantic"] = os.environ.get("TRUSTMEM_SEMANTIC", "1") != "0"
print("semantic", cfg["semantic"], flush=True)
for name in ("hotpot", "2wiki"):
    _, test, pool = load_slices(name, 60, 4)
    print(name, len(test), flush=True)
    evaluate_block(
        models,
        test,
        pool,
        ["crag_plus", "trustmem"],
        ["mixed", "unsupported"],
        cfg,
        n_agents=2,
        mode="random",
        seed=0,
        horizons=(4,),
        label=f"wide-{name}",
    )
print("DONE", flush=True)
