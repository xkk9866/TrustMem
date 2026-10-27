"""Short check that attestation and CRAG run, before the full test split."""

from run_trustmem import (
    FULL_GATES,
    HONEST_CACHE,
    evaluate_split,
    fmt_table,
    load_examples,
    load_models,
)

def main() -> None:
    models = load_models("cuda:0")
    cfg = {"theta_entail": 0.20, "theta_app": 0.12, "theta_q": 0.15, "margin": 0.20, "entail": True}
    HONEST_CACHE.clear()
    dev, _ = load_examples("hotpot", 12, 4, seed=3152)
    result = evaluate_split(
        models,
        dev,
        cfg,
        seeds=[0],
        settings=["clean", "mixed", "conflict", "unsupported"],
        methods=["marag", "crag", "trustmem"],
        gates=FULL_GATES,
    )
    print(fmt_table(result["mean"]), flush=True)
    no_attest = dict(FULL_GATES)
    no_attest["attest"] = False
    HONEST_CACHE.clear()
    ablate = evaluate_split(
        models,
        dev,
        cfg,
        seeds=[0],
        settings=["conflict"],
        methods=["trustmem"],
        gates=no_attest,
    )
    stats = ablate["mean"]["conflict"]["trustmem"]
    print(f"wo_attest conflict F1 {stats['f1']:.3f} pF1 {stats['probe_f1']:.3f} Hold {stats['hold']:.3f}", flush=True)


if __name__ == "__main__":
    main()
