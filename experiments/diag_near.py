"""CPU-only: how often a TrustMem sentence is a near-copy of the corrective read."""

import pickle
from pathlib import Path

from run_trustmem import gold_recall, token_f1

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


def main():
    for dataset in ("hotpot", "2wiki"):
        with (ROOT / "results" / f"packets_{dataset}.pkl").open("rb") as handle:
            packets, _, meta = pickle.load(handle)
        print("\n==", dataset, flush=True)
        for setting in ("mixed", "unsupported", "incomplete", "conflict"):
            n = len(meta)
            gold_m = gold_p = near = only_m = 0
            for index, item in enumerate(meta):
                mem = rows(packets[setting]["trustmem"][index])
                plus = rows(packets[setting]["crag_plus"][index])
                gold = item["answer"]
                gold_m += gold_recall(" ".join(mem), gold) >= 0.99
                gold_p += gold_recall(" ".join(plus), gold) >= 0.99
                hit = False
                for left in mem:
                    if left in plus:
                        continue
                    if any(token_f1(left, right) >= 0.62 for right in plus):
                        hit = True
                        break
                near += hit
                only_m += any(text not in plus for text in mem)
            print(
                f"  {setting:12s} n={n} gold_mem={gold_m/n:.3f} gold_plus={gold_p/n:.3f}"
                f"  near_copy={near/n:.3f} mem_has_extra={only_m/n:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
