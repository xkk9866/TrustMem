"""Dev diagnosis: do retrieved sentences mention the missing gold title?"""

import torch

from run_openrag import first_hop_lists, load_reader, rows_of
from run_trustmem import load_examples, normalize_answer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
models = load_reader(device)
dev, _ = load_examples("hotpot", 36, 16, seed=3152)
both = either = in_rrf = missing = 0
for ex in dev:
    rows = rows_of(ex["docs"])
    lists = first_hop_lists(models, ex["question"], rows, 2)
    gold = [d["title"] for d in ex["docs"] if d["support"]]
    have = set()
    for lst in lists:
        for row in lst:
            have.add(row["title"])
    blobs = []
    for lst in lists:
        blobs.append(normalize_answer(" ".join(f"{r['title']} {r['text']}" for r in lst)))
    union = " ".join(blobs)
    for title in gold:
        if title in have:
            continue
        missing += 1
        needle = normalize_answer(title)
        hit = [needle in blob for blob in blobs]
        if any(hit):
            either += 1
        if all(hit) and len(hit) >= 2:
            both += 1
        if needle in union:
            in_rrf += 1
        print(("BOTH" if all(hit) else "ONE" if any(hit) else "NONE"), title, "| q:", ex["question"][:90])
print("missing", missing, "either", either, "both", both, "union", in_rrf)
