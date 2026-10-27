"""Build the evaluation samples from the cached HotpotQA, 2WikiMultihopQA and MuSiQue files.

Every question type is kept (bridge, comparison, yes/no, 2-4 hop). No answer-span
filter is applied. Questions are shuffled once with a fixed seed and the first N
are written, so a longer run is always a superset of a shorter one.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
OUT = Path(__file__).resolve().parent / (sys.argv[2] if len(sys.argv) > 2 else "data")
SEED = 3152
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def from_context(row) -> list[dict]:
    titles = list(row["context"]["title"])
    groups = list(row["context"]["sentences"])
    docs = []
    for title, sents in zip(titles, groups):
        text = clean(" ".join(clean(s) for s in sents))
        if text:
            docs.append({"title": clean(title), "text": text})
    return docs


def load_hotpot() -> list[dict]:
    path = glob.glob(str(HUB / "datasets--hotpot_qa/snapshots/*/distractor/validation-*.parquet"))[0]
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        items.append(
            {
                "id": str(row["id"]),
                "dataset": "hotpot",
                "question": clean(row["question"]),
                "answer": clean(row["answer"]),
                "aliases": [],
                "type": str(row["type"]),
                "support": sorted(set(str(t) for t in row["supporting_facts"]["title"])),
                "docs": from_context(row),
            }
        )
    return items


def load_2wiki() -> list[dict]:
    path = glob.glob(str(HUB / "datasets--framolfese--2WikiMultihopQA/snapshots/*/data/validation-*.parquet"))[0]
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        items.append(
            {
                "id": str(row["id"]),
                "dataset": "2wiki",
                "question": clean(row["question"]),
                "answer": clean(row["answer"]),
                "aliases": [],
                "type": str(row["type"]),
                "support": sorted(set(str(t) for t in row["supporting_facts"]["title"])),
                "inter": sorted({clean(e[0]) for e in row["evidences"]} | {clean(e[2]) for e in row["evidences"]}),
                "docs": from_context(row),
            }
        )
    return items


def load_musique() -> list[dict]:
    path = glob.glob(str(HUB / "datasets--dgslibisey--MuSiQue/snapshots/*/musique_ans_v1.0_dev.jsonl"))[0]
    items = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if not row.get("answerable", True):
            continue
        docs = [{"title": clean(p["title"]), "text": clean(p["paragraph_text"])} for p in row["paragraphs"]]
        support = sorted({clean(p["title"]) for p in row["paragraphs"] if p.get("is_supporting")})
        items.append(
            {
                "id": str(row["id"]),
                "dataset": "musique",
                "question": clean(row["question"]),
                "answer": clean(row["answer"]),
                "aliases": [clean(a) for a in row.get("answer_aliases") or []],
                "type": row["id"].split("__")[0],
                "support": support,
                "inter": [clean(step["answer"]) for step in row["question_decomposition"]],
                "docs": docs,
            }
        )
    return items


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, loader in (("hotpot", load_hotpot), ("2wiki", load_2wiki), ("musique", load_musique)):
        items = loader()
        rng = random.Random(f"{SEED}:{name}")
        rng.shuffle(items)
        keep = items[:N]
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for item in keep:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        types = {}
        for item in keep:
            types[item["type"]] = types.get(item["type"], 0) + 1
        pages = sum(len(item["docs"]) for item in keep) / max(1, len(keep))
        print(f"{name}: pool {len(items)} kept {len(keep)} pages/q {pages:.1f} types {types}")


if __name__ == "__main__":
    main()
