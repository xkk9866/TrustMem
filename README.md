# TrustMem

TrustMem studies shared memory among agents that hold different source documents. A memory entry is treated as a proposal: agents check its claims against the documents they can access before it influences later reasoning. The implementation includes verification-based methods, memory attacks and comparisons under the same evidence-access rules.

The current pipeline is in `tm/`. Earlier derivation, citation and durable-memory experiments are retained in `experiments/`.

## Code map

| Path | Purpose |
| --- | --- |
| `tm/common.py` | Page assignments, prompts, answer normalization and metrics |
| `tm/methods.py` | TrustMem and comparison methods |
| `tm/engine.py` | Batched generation, scoring and cache handling |
| `tm/run.py` | Dataset, method and attack-condition runner |
| `tm/prep_data.py` | Deterministic benchmark sample preparation |
| `tm/analyze.py`, `tm/final_analysis*.py` | Aggregate and paired analyses of saved runs |
| `experiments/` | Earlier methods, diagnostics and rescoring programs |

## Setup

Use Python 3.10 or newer. Install PyTorch for the target machine, followed by the Python dependencies:

```bash
python -m pip install -r requirements.txt
python tm/run.py --help
```

Model-backed execution also requires vLLM and a compatible CUDA environment. Pass the model directory explicitly with `--model`; the source release does not contain model weights. The runner can batch multiple questions through one model instance.

## Prepare data

Download the HotpotQA, 2WikiMultihopQA and MuSiQue files used by `tm/prep_data.py` into the Hugging Face cache. The preparation script locates the cached files, normalizes their document fields and creates a reproducible shuffled sample:

```bash
python tm/prep_data.py 1000
```

The first argument is the number of examples to prepare. Output is written to `tm/data/` as `hotpot.jsonl`, `2wiki.jsonl` and `musique.jsonl`. The script expects cached inputs and does not download missing datasets itself. Inspect its dataset-specific readers when using a different upstream dataset revision.

## Run an experiment

After preparing the data and model, run clean and attacked conditions with the same question set:

```bash
python tm/run.py --datasets hotpot,2wiki,musique --n 500 --methods trustmem --conditions clean,poison1 --model models/Qwen2.5-7B-Instruct-AWQ --tag main --no-wait
```

Use `--methods` and `--conditions` to select the comparisons implemented in `tm/methods.py`. `--n`, `--offset` and `--seed` control the evaluation slice and randomization. The output directory includes the selected run tag; preserve the run configuration together with prediction records when comparing experiments.

For pipeline development, `--mock` selects the mock engine and avoids loading a language model. Mock outputs test execution and bookkeeping, not model quality. The `--serve` mode processes local JSON queue specifications with one resident engine; it does not require publishing a network service.

## Release contents

The repository includes the available implementation and experiment source. Historical diagnostics in `experiments/` may require their corresponding saved predictions or cached intermediate files. The current `tm/` pipeline is the starting point for new runs.

Manuscripts, submission documents, publication figures, generated run logs and private connection files are excluded. Public benchmark data should be obtained from its original providers. `RELEASE_MANIFEST.json` records the released source files and hashes.
