"""Dev diagnosis for attested title links. Does not score the test split."""

import torch

from run_openrag import first_hop_lists, load_reader, rows_of
from run_trustmem import load_examples, normalize_answer, tokens


def link_titles(question: str, docs: list, lists: list, need: int) -> list[str]:
    selected = {row["title"] for lst in lists for row in lst}
    q_words = set(normalize_answer(question).split())
    blobs = [normalize_answer(" ".join(f"{r['title']} {r['text']}" for r in lst)) for lst in lists]
    hits = []
    for doc in docs:
        title = doc["title"]
        if title in selected:
            continue
        needle = normalize_answer(title)
        if len(needle) < 3:
            continue
        words = [w for w in needle.split() if w not in {"a", "an", "the"}]
        if words and all(w in q_words for w in words):
            continue
        agents = sum(needle in blob for blob in blobs)
        if agents >= need:
            hits.append((agents, len(needle), title))
    hits.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [title for _, _, title in hits]


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = load_reader(device)
    for dataset in ("hotpot", "2wiki"):
        dev, _ = load_examples(dataset, 36, 16, seed=3152)
        for need, name in ((2, "both"), (1, "either")):
            fired = gold_hit = gold_miss = false = 0
            for ex in dev:
                rows = rows_of(ex["docs"])
                lists = first_hop_lists(models, ex["question"], rows, 2)
                gold = {d["title"] for d in ex["docs"] if d["support"]}
                have = {row["title"] for lst in lists for row in lst}
                missing = gold - have
                picked = link_titles(ex["question"], ex["docs"], lists, need)
                if not picked:
                    gold_miss += len(missing)
                    continue
                fired += 1
                choice = picked[0]
                if choice in gold:
                    gold_hit += 1
                else:
                    false += 1
                still = missing - {choice}
                gold_miss += len(still)
            print(dataset, name, "fired", fired, "gold", gold_hit, "false", false, "miss_left", gold_miss, "n", len(dev))


if __name__ == "__main__":
    main()
