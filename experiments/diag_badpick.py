import re

from diag_compare2 import pick
from run_durable import load_slices

for dataset in ("hotpot", "2wiki"):
    _, test, _ = load_slices(dataset, 120, 4)
    print("==", dataset)
    for item in test:
        answer = pick(item["question"], item["docs"])
        if not answer:
            continue
        mark = "OK" if answer.lower() == item["answer"].lower() else "BAD"
        if dataset == "hotpot" or mark == "BAD":
            print(mark, item["answer"], "||", answer, "||", item["question"])
