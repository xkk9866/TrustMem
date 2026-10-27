"""Pilot of faithful provenance and question-bound readout."""
from run_durable import base_cfg, evaluate_block, load_slices
from run_trustmem import load_models

models = load_models("cuda:1")
_, test, pool = load_slices("hotpot", 60, 100)
print("lens", len(test), len(pool), flush=True)
summary, _ = evaluate_block(
    models,
    test,
    pool,
    ["marag", "crag", "crag_plus", "trustmem"],
    ["mixed", "unsupported", "incomplete", "conflict", "paraphrase"],
    base_cfg(),
    n_agents=2,
    mode="random",
    seed=0,
    horizons=(4, 100),
    label="pilot",
)
print("DONE", flush=True)
