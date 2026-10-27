"""Does the frozen reader find the answer when it sees the whole admitted packet?"""
import pickle
from pathlib import Path

from run_durable import LIVE
from run_trustmem import Contract, load_models, n_words, reader_from_contracts, token_f1, tokens

root = Path(__file__).resolve().parents[1]
models = load_models("cuda:1")
with (root / "results" / "packets_pilot.pkl").open("rb") as handle:
    test_packets, _, meta = pickle.load(handle)


def all_evs(stored):
    rows = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or [{"text": contract.claim, "title": ""}]:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                rows.append(ev)
    return rows


def bridge_first(question, stored):
    rows = all_evs(stored)
    titles = [ev.get("title", "") for ev in rows if ev.get("title")]
    q = tokens(question)

    def mentions_other(ev):
        text_toks = tokens(ev.get("text", ""))
        own = tokens(ev.get("title", ""))
        for title in titles:
            other = tokens(title)
            if other and other != own and other <= text_toks:
                return True
        return False

    linked = [ev for ev in rows if mentions_other(ev)]
    named = set()
    for ev in linked:
        text_toks = tokens(ev.get("text", ""))
        for title in titles:
            other = tokens(title)
            if other and other <= text_toks:
                named.add(title)
    leads = []
    for title in named:
        page = [ev for ev in rows if ev.get("title") == title]
        page.sort(key=lambda ev: len(q & tokens(ev.get("text", ""))), reverse=True)
        if page:
            leads.append(page[0])
    rest = sorted(rows, key=lambda ev: len(q & tokens(f"{ev.get('title', '')} {ev.get('text', '')}")), reverse=True)
    ordered = []
    seen = set()
    for ev in linked + leads + rest:
        if ev.get("text") not in seen:
            seen.add(ev.get("text"))
            ordered.append(ev)
    return ordered


def f1_context(question, answer, evs):
    packed = Contract("", evs, "readout", "read", 0.5, 0, LIVE, "none", False)
    pred, _, _ = reader_from_contracts(models, question, [packed])
    return token_f1(pred, answer)


for setting in ("unsupported", "mixed", "incomplete", "conflict"):
    print("\n==", setting, flush=True)
    for method in ("crag_plus", "trustmem"):
        scores = {"top4": [], "all": [], "bridge": []}
        for index, stored in enumerate(test_packets[setting][method]):
            question = meta[index]["question"]
            answer = meta[index]["answer"]
            evs = all_evs(stored)
            q = tokens(question)
            ranked = sorted(evs, key=lambda ev: len(q & tokens(ev.get("text", ""))), reverse=True)
            scores["top4"].append(f1_context(question, answer, ranked[:4]))
            scores["all"].append(f1_context(question, answer, evs))
            scores["bridge"].append(f1_context(question, answer, bridge_first(question, stored)[:6]))
        msg = " ".join(f"{name}={sum(vals)/len(vals):.3f}" for name, vals in scores.items())
        print(f"  {method:10s} {msg}", flush=True)
