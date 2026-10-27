"""Development-slice probe. Does not score the reported test slice."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from derivation import derive
from run_durable import load_slices
from run_sota import load_reader, read_page
from run_trustmem import token_f1

reader = load_reader(os.environ.get("TRUSTMEM_DEVICE", "cuda:1"))


def read(question, doc):
    return read_page(reader, question, doc), 0.0


for dataset in ("2wiki", "hotpot"):
    dev, _, _ = load_slices(dataset, 120, 4)
    fired = []
    kinds = {}
    print(f"\n== dev {dataset} n={len(dev)}", flush=True)
    for item in dev:
        value, kind, pages = derive(item["question"], item["docs"], read)
        kind = kind or "packet"
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "packet":
            continue
        f1 = token_f1(value, item["answer"])
        fired.append(f1)
        if f1 < 0.5:
            print(
                f"  miss {kind} f1={f1:.2f} pred={value!r} gold={item['answer']!r}",
                flush=True,
            )
            print(f"    Q {item['question']}", flush=True)
    fired_mean = sum(fired) / len(fired) if fired else 0.0
    print(f"fired {len(fired)} fired_mean {fired_mean:.3f} kinds {kinds}", flush=True)
