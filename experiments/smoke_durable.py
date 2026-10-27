"""Four-question check that durable admission and readout run."""
from run_durable import base_cfg, evaluate_block, load_models, load_slices

models = load_models("cuda:0")
dev, test, pool = load_slices("hotpot", 4, 8)
print("lens", len(test), len(pool))
summary, _ = evaluate_block(
    models,
    test,
    pool,
    ["marag", "crag", "crag_plus", "trustmem"],
    ["clean", "paraphrase", "conflict"],
    base_cfg(),
    n_agents=2,
    mode="random",
    seed=0,
    horizons=(4,),
    label="smoke",
)
print(summary)
