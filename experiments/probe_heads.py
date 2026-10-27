"""Show the phrase reduction on the development slice. No reader."""
from denote import containing_phrase, head_word, inner_question, named_pages
from diag_compare2 import choices_of
from run_durable import load_slices

for dataset in ("2wiki", "hotpot"):
    dev, _, _ = load_slices(dataset, 120, 4)
    named_n = order_n = head_n = 0
    for item in dev:
        question = item["question"]
        if len(choices_of(question)) == 2:
            order_n += 1
            continue
        named = named_pages(question, item["docs"])
        if not named:
            continue
        named_n += 1
        phrase = containing_phrase(question, named[0]["title"])
        head = head_word(phrase, named[0]["title"])
        if head:
            head_n += 1
        else:
            print(dataset, "NOHEAD", question[:110])
    print(dataset, "dev", len(dev), "order", order_n, "named", named_n, "head", head_n)
