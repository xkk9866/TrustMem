from datasets import load_dataset

target = "5ae2e73a55429928c4239545"
raw = load_dataset("hotpot_qa", "distractor", split="validation")
for ex in raw:
    if ex["id"] != target:
        continue
    print("Q:", ex["question"])
    print("A:", ex["answer"])
    titles = ex["context"]["title"]
    sents = ex["context"]["sentences"]
    support = set(ex["supporting_facts"]["title"])
    for title, group in zip(titles, sents):
        if title not in support:
            continue
        print("PAGE", title)
        for sent in group:
            if "Gump" in sent or "nickname" in sent.lower() or "Forrest" in sent:
                print(" ", sent)
    break
