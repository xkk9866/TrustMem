"""Dev-only diagnosis of when the residual hop fires. Not a test evaluation."""

from run_openrag import (
    answer_method,
    first_hop_lists,
    load_reader,
    place_new_page,
    rrf_fuse,
    rows_of,
    second_hop,
    shared_entity,
)
from run_trustmem import load_examples
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
models = load_reader(device)
dev, _ = load_examples("hotpot", 36, 16, seed=3152)
n_share = n_new = n_already = 0
for ex in dev[:36]:
    rows = rows_of(ex["docs"])
    lists = first_hop_lists(models, ex["question"], rows, 2)
    bridge = shared_entity(ex["question"], lists)
    base = rrf_fuse(lists, k=4)
    gold = [d["title"] for d in ex["docs"] if d["support"]]
    have = {r["title"] for r in base}
    if not bridge:
        print("NO", gold, "have", sorted(have)[:4])
        continue
    n_share += 1
    hop2 = second_hop(models, bridge, rows, 2, bind=True)
    fused, fired = place_new_page(base, hop2, 1)
    new_titles = [r["title"] for r in hop2 if r["title"] not in have]
    if fired:
        n_new += 1
    else:
        n_already += 1
    print(
        "BRIDGE" if fired else "HAVE",
        "bridge=",
        bridge[:60],
        "gold=",
        gold,
        "new=",
        new_titles[:2],
        "have=",
        list(have)[:3],
    )
print("share", n_share, "new", n_new, "already", n_already, "of", 36)
