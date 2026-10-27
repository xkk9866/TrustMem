"""2Wiki items whose answer sentence is only in the attested log."""

import pickle
from pathlib import Path

from run_trustmem import normalize_answer

ROOT = Path(__file__).resolve().parents[1]


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append((ev.get("title") or "", text))
    return out


def carries(text, gold):
    needle = normalize_answer(gold)
    return bool(needle) and needle not in {"yes", "no"} and needle in normalize_answer(text)


def main():
    with (ROOT / "results" / "packets_2wiki.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    shown = 0
    for index, item in enumerate(meta):
        gold = item["answer"]
        mem = rows(packets["mixed"]["trustmem"][index])
        plus = rows(packets["mixed"]["crag_plus"][index])
        plus_text = {text for _, text in plus}
        exclusive = [(title, text) for title, text in mem if carries(text, gold) and text not in plus_text]
        if not exclusive:
            continue
        print("\nQ:", item["question"])
        print("A:", gold)
        print("-- corrective")
        for title, text in plus:
            mark = " *" if carries(text, gold) else ""
            print(f"  [{title[:42]}]{mark} {text[:160]}")
        print("-- answer sentence absent from corrective")
        for title, text in exclusive:
            print(f"  [{title[:42]}] {text[:200]}")
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()
