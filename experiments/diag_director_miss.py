"""Inspect the one younger-director miss and title-anchor mention coverage."""

from diag_compare2 import birth_only, director_doc, find_doc, pick
from diag_bridge_cover import anchor, mentioned
from run_durable import load_slices
from run_trustmem import normalize_answer


def main():
    _, test, _ = load_slices("2wiki", 120, 4)
    for item in test:
        if "Deadly Eyes" not in item["question"]:
            continue
        docs = item["docs"]
        print("Q", item["question"])
        print("gold", item["answer"], "pick", pick(item["question"], docs))
        opts = ["I Was A Student At Heidelberg", "Deadly Eyes"]
        for name in opts:
            film = find_doc(docs, name)
            person = director_doc(docs, film)
            print("---", name)
            print(" film", None if film is None else film["title"])
            if film:
                print(" lede", film["sentences"][0][:240])
            print(" director", None if person is None else person["title"], "birth", birth_only(person))
            if person:
                print(" dlede", person["sentences"][0][:240])

    _, hot, _ = load_slices("hotpot", 120, 4)
    on_anchor = on_link = on_either = has_link = 0
    for item in hot:
        page = anchor(item["question"], item["docs"])
        links = mentioned(page, item["docs"]) if page is not None else []
        ans = normalize_answer(item["answer"])
        def has(doc):
            return doc is not None and ans in normalize_answer(" ".join(doc["sentences"]))
        a, b = has(page), any(has(doc) for doc in links)
        on_anchor += a
        on_link += b
        on_either += a or b
        has_link += bool(links)
    print(f"hotpot anchor {on_anchor} link {on_link} either {on_either} with_link {has_link} n {len(hot)}")


if __name__ == "__main__":
    main()
