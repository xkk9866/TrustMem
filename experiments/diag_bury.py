"""Where the answer sentence sits: attested log, corrective read, or both."""

import pickle
from pathlib import Path

from run_trustmem import normalize_answer, tokens

ROOT = Path(__file__).resolve().parents[1]


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text") or ""
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def carries(text, gold):
    needle = normalize_answer(gold)
    if not needle or needle in {"yes", "no"}:
        return False
    return needle in normalize_answer(text)


def overlap(question, text):
    q = tokens(question)
    if not q:
        return 0.0
    return len(q & tokens(text)) / len(q)


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "clean", "conflict"):
            n = len(meta)
            in_mem = in_plus = in_both = both_outside_top = 0
            for item, index in zip(meta, range(n)):
                question = item["question"]
                gold = item["answer"]
                mem = rows(packets[setting]["trustmem"][index])
                plus = rows(packets[setting]["crag_plus"][index])
                mem_hit = [text for text in mem if carries(text, gold)]
                plus_hit = [text for text in plus if carries(text, gold)]
                in_mem += bool(mem_hit)
                in_plus += bool(plus_hit)
                both = [text for text in mem_hit if text in plus]
                in_both += bool(both)
                ranked = sorted(mem, key=lambda text: overlap(question, text), reverse=True)[:4]
                both_outside_top += bool(both) and all(text not in ranked for text in both)
            print(
                f"  {setting:12s} in_mem={in_mem/n:.3f} in_plus={in_plus/n:.3f}"
                f"  in_both={in_both/n:.3f} both_buried={both_outside_top/n:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
