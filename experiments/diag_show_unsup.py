"""Show a few unsupported packets: attested sentences versus the corrective read."""

import pickle
from pathlib import Path

from run_trustmem import normalize_answer, token_f1

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
    with (ROOT / "results" / "packets_hotpot.pkl").open("rb") as handle:
        packets, _, meta = pickle.load(handle)
    shown = 0
    for index, item in enumerate(meta):
        gold = item["answer"]
        mem = rows(packets["unsupported"]["trustmem"][index])
        plus = rows(packets["unsupported"]["crag_plus"][index])
        plus_text = {text for _, text in plus}
        if not any(carries(text, gold) for _, text in mem):
            continue
        extra = [(title, text) for title, text in mem if text not in plus_text]
        if not extra:
            continue
        print("\nQ:", item["question"])
        print("A:", gold)
        print("-- corrective")
        for title, text in plus:
            mark = " *" if carries(text, gold) else ""
            print(f"  [{title[:40]}]{mark} {text[:180]}")
        print("-- attested only")
        for title, text in extra:
            mark = " *" if carries(text, gold) else ""
            best = max((token_f1(text, other) for _, other in plus), default=0.0)
            print(f"  [{title[:40]}]{mark} near={best:.2f} {text[:180]}")
        shown += 1
        if shown >= 4:
            break


if __name__ == "__main__":
    main()
