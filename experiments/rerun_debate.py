"""Replace the debate column after the self-support rule change."""

import json

from run_trustmem import (
    FAULTS,
    FULL_GATES,
    HONEST_CACHE,
    RESULT_PATH,
    ROOT,
    evaluate_split,
    fmt_table,
    load_examples,
    load_models,
)


def main() -> None:
    path = ROOT / "results" / "closure_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    models = load_models("cuda:0")
    cfg = {
        "theta_entail": 0.20,
        "theta_app": 0.12,
        "theta_q": 0.15,
        "margin": 0.20,
        "entail": True,
    }
    for name in ("hotpot", "2wiki"):
        print("dataset", name, flush=True)
        HONEST_CACHE.clear()
        _, test = load_examples(name, 36, 120, seed=3152)
        result = evaluate_split(
            models,
            test,
            cfg,
            seeds=[0, 1, 2],
            settings=["clean", "mixed", *FAULTS],
            methods=["debate"],
            gates=FULL_GATES,
        )
        block = data["datasets"][name]
        for setting, methods in result["mean"].items():
            block["mean"][setting]["debate"] = methods["debate"]
        for seed, settings in result["seeds"].items():
            for setting, methods in settings.items():
                block["seeds"][seed][setting]["debate"] = methods["debate"]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(fmt_table(block["mean"]), flush=True)
    print("wrote", path, flush=True)
    del RESULT_PATH


if __name__ == "__main__":
    main()
