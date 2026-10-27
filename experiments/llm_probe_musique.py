import os

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset

for split in ("validation", "train"):
    try:
        ds = load_dataset("dgslibisey/MuSiQue", split=split)
    except Exception as exc:
        print(split, "fail", exc)
        continue
    print(split, len(ds), ds.column_names)
    ex = ds[0]
    for key, value in ex.items():
        print(" ", key, repr(value)[:300])
    print("answerable", sum(ds["answerable"]) if "answerable" in ds.column_names else "na")
    break
