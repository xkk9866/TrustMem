"""Durable readout: a long log, matched read time, and a short reader budget.

Every method stores what its write rule accepts and later reads the same way:
rank stored contracts by applicability to the question being asked, then pass
four sentences to the shared reader. Horizons of 4, 100, and 1000 later
questions use distinct interference questions, not a second pass over the
eight-slot buffer.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np

from run_trustmem import (
    FULL_GATES,
    Contract,
    Models,
    admit_trustmem,
    answer_supported,
    apply_fault,
    assign_strength,
    bm25_sentences,
    build_fault_contract,
    citation_hop,
    clone_contract,
    close_contracts,
    crag_select,
    doc_index,
    gold_recall,
    grounded,
    load_models,
    make_honest_contract,
    marag_select,
    merge_replies,
    parse_example,
    peer_attest,
    prov_ok,
    reader_from_contracts,
    token_f1,
    tokens,
    top_paragraphs,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "durable_results.json"
LIVE = 10**9
HORIZONS = (4, 100, 1000)
FAULTS = ["unsupported", "incomplete", "stale", "conflict", "paraphrase"]


def load_slices(dataset: str, n_test: int, n_pool: int, seed: int = 3152):
    from datasets import load_dataset

    if dataset == "hotpot":
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
    else:
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    parsed = []
    for ex in raw:
        item = parse_example(ex)
        if dataset == "hotpot" and item["type"] != "bridge":
            continue
        if not answer_supported(item):
            continue
        parsed.append(item)
    rng = random.Random(seed)
    rng.shuffle(parsed)
    test = parsed[36 : 36 + n_test]
    pool = parsed[36 + n_test : 36 + n_test + n_pool]
    dev = parsed[:36]
    return dev, test, pool


def partition(docs: list, n_agents: int, mode: str, rng: random.Random) -> list[list]:
    groups = [[] for _ in range(n_agents)]
    if mode == "assigned":
        support = [doc for doc in docs if doc["support"]]
        other = [doc for doc in docs if not doc["support"]]
        ordered = support + other
    elif mode == "random":
        ordered = list(docs)
        rng.shuffle(ordered)
    elif mode == "retrieve":
        return [list(docs) for _ in range(n_agents)]
    else:
        raise ValueError(mode)
    for index, doc in enumerate(ordered):
        groups[index % n_agents].append(doc)
    return groups


def contract_at_rank(models: Models, question: str, docs: list, producer: str, qid: str, t: int, rank: int) -> Contract:
    paras = top_paragraphs(question, docs, k=max(rank + 1, 1))
    chosen = paras[rank : rank + 1] or paras[:1]
    evidence = bm25_sentences(question, chosen, k=4)
    context = "\n".join(f"{ev['title']}: {ev['text']}" for ev in evidence)
    claim, score = models.answer(question, context)
    return Contract(
        claim=claim,
        evidence=evidence,
        producer=producer,
        query_id=qid,
        confidence=float(min(score, 0.95)),
        t_start=t,
        t_end=LIVE,
        fault="none",
        injected=False,
        owned=tuple(doc["doc_id"] for doc in chosen),
    )


def widen(contracts: list[Contract]) -> None:
    for contract in contracts:
        if contract.fault != "stale":
            contract.t_end = LIVE


HONEST = {}


def honest_peers(models, ex, n_agents, mode, t, qid):
    key = (str(qid), n_agents, mode)
    cached = HONEST.get(key)
    if cached is None:
        rng = random.Random(f"{qid}:{n_agents}:{mode}")
        groups = partition(ex["docs"], n_agents, mode, rng)
        peers = []
        for index, group in enumerate(groups):
            if not group:
                continue
            producer = f"agent_{index}"
            if mode == "retrieve":
                peers.append(contract_at_rank(models, ex["question"], group, producer, qid, t, index))
            else:
                peers.append(make_honest_contract(models, ex["question"], group, producer, qid, t))
        widen(peers)
        HONEST[key] = peers
    return [clone_contract(contract) for contract in HONEST[key]]


def prepare_peers(models, ex, fault, rng, n_agents, mode, t, qid):
    peers = honest_peers(models, ex, n_agents, mode, t, qid)
    fault_contract = None
    extra_docs = []
    if fault != "none":
        fault_contract, extra_docs = build_fault_contract(
            models, ex["question"], ex["docs"], peers, fault, qid, t, rng, ex["answer"]
        )
    if fault in {"unsupported", "incomplete", "paraphrase"} and fault_contract is not None:
        replaced = False
        out = []
        for peer in peers:
            if not replaced and peer.producer == fault_contract.producer:
                out.append(fault_contract)
                replaced = True
            else:
                out.append(peer)
        peers = out
    elif fault_contract is not None:
        peers = list(peers) + [fault_contract]
    widen(peers)
    docs = doc_index(list(ex["docs"]) + list(extra_docs))
    # A closed window is ineligible for every method, including the baselines.
    peers = [contract for contract in peers if contract.t_start <= t <= contract.t_end]
    return peers, docs


def crag_plus_select(peers, question, docs, qid, allow_expand):
    """CRAG, plus a refetch when a verbatim quote does not contain the claim."""
    from run_trustmem import cited_docs

    fresh = [contract for contract in peers if contract.query_id == qid] or list(peers)
    collected = []
    for contract in fresh:
        on_page = bool(contract.evidence) and prov_ok(contract, docs)
        claim_gap = bool(contract.claim) and not grounded(contract.claim, contract.evidence_text())
        if allow_expand and contract.evidence and ((not on_page) or claim_gap):
            retrieved = bm25_sentences(question, cited_docs(contract, docs), k=2)
            if retrieved:
                collected.extend(retrieved)
                continue
        if contract.evidence:
            collected.extend(contract.evidence[:2])
        elif contract.claim:
            collected.append({"doc_id": "", "sent_id": 0, "title": "message", "text": contract.claim})
    if not collected:
        return []
    pseudo = [
        {"doc_id": f"plus:{i}", "title": ev.get("title", ""), "sentences": [ev["text"]], "support": False}
        for i, ev in enumerate(collected)
        if ev.get("text")
    ]
    evidence = bm25_sentences(question, pseudo, k=4)
    restored = []
    for ev in evidence:
        source = next((item for item in collected if item.get("text") == ev["text"]), ev)
        restored.append(
            {
                "doc_id": source.get("doc_id", ""),
                "sent_id": source.get("sent_id", 0),
                "title": source.get("title", ""),
                "text": ev["text"],
            }
        )
    window = fresh[0]
    return [
        Contract(
            claim="",
            evidence=restored,
            producer="crag_plus",
            query_id=qid,
            confidence=0.55,
            t_start=window.t_start,
            t_end=LIVE,
            fault="none",
            injected=False,
            strength=0.55,
        )
    ]


def admit_method(method, models, peers, question, t, docs, cfg, qid):
    if method == "trustmem" and os.environ.get("TRUSTMEM_PACKET") != "cite":
        admitted, _ = admit_trustmem(models, peers, question, t, docs, cfg, True)
        if cfg.get("attest", True):
            replies = peer_attest(admitted, question, docs)
            admitted = merge_replies(admitted, replies)
        assign_strength(models, admitted, question, t, docs)
        closed, _ = close_contracts(models, admitted, question, cfg, limit=32)
        stored = closed or admitted
    elif method == "trustmem":
        # Same tight page read as CRAG+. The extra step opens a page whose title that read names, and keeps the naming sentence.
        admitted, _ = admit_trustmem(models, peers, question, t, docs, cfg, True)
        replies = citation_hop(admitted, question, docs) if cfg.get("attest", True) else []
        base_list = crag_plus_select(peers, question, docs, qid, True)
        base_ev = list(base_list[0].evidence) if base_list else []
        window = base_list[0] if base_list else (admitted[0] if admitted else peers[0])
        extra = []
        seen = {ev.get("text") for ev in base_ev}
        for reply in replies:
            for ev in reply.evidence:
                title = (ev.get("title") or "").strip()
                text = ev.get("text") or ""
                if text and text not in seen and len(title.split()) >= 2:
                    extra.append(ev)
                    seen.add(text)
        extra = extra[:2]
        if not extra:
            evidence = base_ev
        else:
            extra_titles = [(ev.get("title") or "").strip() for ev in extra]
            linked = []
            for ev in base_ev:
                text = (ev.get("text") or "").lower()
                title = (ev.get("title") or "").strip()
                names_hop = any(len(item) > 3 and item.lower() in text for item in extra_titles)
                named_by_hop = any(
                    len(title) > 3 and title.lower() in (item.get("text") or "").lower() for item in extra
                )
                if names_hop or named_by_hop:
                    linked.append(ev)
            evidence = (linked + extra) if linked else (base_ev[:2] + extra)
            evidence = evidence[:4]
        stored = [
            Contract(
                claim="",
                evidence=evidence,
                producer="trustmem",
                query_id=qid,
                confidence=0.55,
                t_start=window.t_start,
                t_end=LIVE,
                fault="none",
                injected=False,
                strength=0.55,
            )
        ] if evidence else []
    elif method == "crag":
        stored, _ = crag_select(peers, question, docs, qid, True)
    elif method == "crag_plus":
        stored = crag_plus_select(peers, question, docs, qid, True)
    elif method == "marag":
        stored = marag_select(question, peers, qid)
    else:
        raise ValueError(method)
    qvec = models.embed(question, query=True)
    for contract in stored:
        if contract.fault != "stale":
            contract.t_end = LIVE
        text = contract.evidence_text() or contract.claim or "empty"
        contract.vec = models.embed(text, query=False)
        contract.write_vec = qvec
    return stored


def applicability(models: Models, question: str, contract: Contract, qvec: np.ndarray) -> float:
    q_toks = tokens(question)
    blob = contract.evidence_text() or contract.claim
    b_toks = tokens(blob)
    overlap = (len(q_toks & b_toks) / len(q_toks)) if q_toks else 0.0
    vec = getattr(contract, "vec", None)
    if vec is None:
        vec = models.embed(blob or "empty", query=False)
        contract.vec = vec
    cos = float(np.dot(qvec, vec))
    return 0.45 * overlap + 0.55 * max(cos, 0.0)


def readout(models: Models, question: str, memory: list[Contract], t: int) -> tuple[str, str]:
    alive = [
        contract
        for contract in memory
        if contract.t_start <= t <= contract.t_end and (contract.evidence or contract.claim)
    ]
    if not alive:
        return "", ""
    qvec = models.embed(question, query=True)
    q_toks = tokens(question)
    scored = []
    seen = set()
    for contract in alive:
        evidence = contract.evidence or [{"doc_id": "", "sent_id": 0, "title": "message", "text": contract.claim}]
        for ev in evidence:
            text = ev.get("text", "")
            if not text or text in seen:
                continue
            seen.add(text)
            overlap = (len(q_toks & tokens(text)) / len(q_toks)) if q_toks else 0.0
            vec = models.embed(text, query=False)
            cos = float(np.dot(qvec, vec))
            base = 0.45 * overlap + 0.55 * max(cos, 0.0)
            write_vec = getattr(contract, "write_vec", None)
            if write_vec is not None and os.environ.get("TRUSTMEM_BIND", "1") != "0":
                bind = 0.55 + 0.45 * max(float(np.dot(qvec, write_vec)), 0.0)
                base *= bind
            scored.append((base, ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    sentences = [ev for _, ev in scored[:4]]
    packed = Contract(
        claim="",
        evidence=sentences,
        producer="readout",
        query_id="read",
        confidence=0.5,
        t_start=0,
        t_end=LIVE,
        fault="none",
        injected=False,
    )
    try:
        pred, _, _ = reader_from_contracts(models, question, [packed])
    except RuntimeError:
        import torch

        torch.cuda.empty_cache()
        pred, _, _ = reader_from_contracts(models, question, [packed])
    blob = " ".join(ev["text"] for ev in sentences)
    return pred, blob


def base_cfg(theta_entail: float = 0.20, semantic_tau: float = 0.5) -> dict:
    cfg = dict(FULL_GATES)
    cfg.update(
        {
            "theta_entail": theta_entail,
            "theta_app": 0.0,
            "theta_q": 0.15,
            "margin": 0.20,
            "entail": True,
            "qscore": False,
            "semantic": True,
            "semantic_tau": semantic_tau,
            "attest": True,
            "attest_pair": True,
        }
    )
    return cfg


def fault_of(setting: str, index: int) -> str:
    if setting == "clean":
        return "none"
    if setting == "mixed":
        return ["unsupported", "incomplete", "stale", "conflict"][index % 4]
    return setting


def write_stream(models, examples, methods, settings, cfg, n_agents, mode, seed, label):
    packed = {setting: {method: [] for method in methods} for setting in settings}
    meta = []
    for index, ex in enumerate(examples):
        t = 10 + index
        qid = str(ex["id"])
        prepared = {}
        for setting in settings:
            fault = fault_of(setting, index)
            rng_fault = random.Random(f"{seed}:{qid}:{setting}")
            peers, docs = prepare_peers(models, ex, fault, rng_fault, n_agents, mode, t, qid)
            prepared[setting] = (peers, docs)
        meta.append({"question": ex["question"], "answer": ex["answer"], "t": t, "id": qid})
        if "clean" in prepared:
            clean_blob = " ".join(contract.evidence_text() for contract in prepared["clean"][0])
            meta[-1]["retrieved"] = gold_recall(clean_blob, ex["answer"]) > 0
        else:
            meta[-1]["retrieved"] = True
        for setting in settings:
            peers, docs = prepared[setting]
            for method in methods:
                stored = admit_method(
                    method, models, [clone_contract(c) for c in peers], ex["question"], t, docs, cfg, qid
                )
                packed[setting][method].append(stored)
        if (index + 1) % 40 == 0:
            print(f"  {label} {index + 1}/{len(examples)}", flush=True)
    return packed, meta


def score_item(models, question, answer, own, interference, horizons):
    pred, blob = readout(models, question, own, 10**6)
    immediate = token_f1(pred, answer)
    delayed = {}
    hold = {}
    mem = list(own)
    cursor = 0
    for horizon in horizons:
        if horizon > len(interference):
            continue
        while cursor < horizon:
            mem.extend(interference[cursor])
            cursor += 1
        pred_h, blob_h = readout(models, question, mem, 10**6)
        delayed[str(horizon)] = token_f1(pred_h, answer)
        hold[str(horizon)] = gold_recall(blob_h, answer)
    return immediate, delayed, hold


def mean_pack(rows):
    if not rows:
        return {}
    keys_d = rows[0]["delayed"].keys()
    out = {
        "n": len(rows),
        "f1": float(np.mean([row["f1"] for row in rows])),
    }
    out["delayed"] = {key: float(np.mean([row["delayed"][key] for row in rows])) for key in keys_d}
    out["hold"] = {key: float(np.mean([row["hold"][key] for row in rows])) for key in keys_d}
    out["f1s"] = [row["f1"] for row in rows]
    out["delayed_f1s"] = {key: [row["delayed"][key] for row in rows] for key in keys_d}
    return out


def bootstrap_delta(left, right, n_boot: int = 2000, seed: int = 3152):
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"delta": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(diff))}


def evaluate_block(models, test, pool, methods, settings, cfg, n_agents, mode, seed, horizons, label):
    safe = label.replace("/", "_")
    ckpt = ROOT / "results" / f"packets_{safe}.pkl"
    if ckpt.exists() and os.environ.get("DURABLE_REUSE") == "1":
        with ckpt.open("rb") as handle:
            test_packets, pool_packets, meta = pickle.load(handle)
        print(f"reuse {ckpt}", flush=True)
    else:
        print(f"write {label} test {len(test)} pool {len(pool)}", flush=True)
        test_packets, meta = write_stream(models, test, methods, settings, cfg, n_agents, mode, seed, label + "/test")
        pool_packets, _ = write_stream(models, pool, methods, ["clean"], cfg, n_agents, mode, seed + 99, label + "/pool")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        with ckpt.open("wb") as handle:
            pickle.dump((test_packets, pool_packets, meta), handle)
        print(f"checkpoint {ckpt}", flush=True)
    summary = {}
    details = {}
    score_path = ROOT / "results" / f"scores_{safe}.json"
    saved = {}
    if score_path.exists() and os.environ.get("DURABLE_REUSE") == "1":
        saved = json.loads(score_path.read_text(encoding="utf-8"))
    for setting in settings:
        summary[setting] = {}
        details[setting] = {}
        for method in methods:
            key = f"{setting}:{method}"
            if key in saved:
                packed = saved[key]
                rows = []
            else:
                rows = []
                interference = pool_packets["clean"][method]
                for index, stored in enumerate(test_packets[setting][method]):
                    start = index % max(len(interference), 1)
                    ordered = interference[start:] + interference[:start]
                    immediate, delayed, hold = score_item(
                        models, meta[index]["question"], meta[index]["answer"], stored, ordered, horizons
                    )
                    rows.append(
                        {
                            "f1": immediate,
                            "delayed": delayed,
                            "hold": hold,
                            "retrieved": bool(meta[index].get("retrieved", True)),
                        }
                    )
                packed = mean_pack(rows)
                missed = [row for row in rows if not row["retrieved"]]
                if missed:
                    packed["miss_n"] = len(missed)
                    packed["miss_f1"] = float(np.mean([row["f1"] for row in missed]))
                saved[key] = packed
                score_path.write_text(json.dumps(saved), encoding="utf-8")
                import torch

                torch.cuda.empty_cache()
            summary[setting][method] = {
                key: packed[key] for key in ("n", "f1", "delayed", "hold") if key in packed
            }
            if "miss_n" in packed:
                summary[setting][method]["miss_n"] = packed["miss_n"]
                summary[setting][method]["miss_f1"] = packed["miss_f1"]
            details[setting][method] = packed
            delayed_txt = " ".join(f"H{key}={packed['delayed'][key]:.3f}" for key in packed["delayed"])
            print(f"  {setting:12s} {method:10s} F1 {packed['f1']:.3f} {delayed_txt}", flush=True)
    return summary, details


def main():
    import torch

    device = os.environ.get("DURABLE_DEVICE", "cuda:0")
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.45, int(device.split(":")[1]))
    models = load_models(device)
    cfg = base_cfg()
    methods = ["marag", "crag", "crag_plus", "trustmem"]
    settings = ["clean", "mixed", "unsupported", "incomplete", "stale", "conflict", "paraphrase"]
    output = {"config": {key: cfg[key] for key in ("theta_entail", "theta_app", "semantic_tau", "qscore")}, "horizons": list(HORIZONS)}
    output["datasets"] = {}
    details = {}
    for name, n_pool in (("hotpot", 1000), ("2wiki", 1000)):
        dev, test, pool = load_slices(name, 120, n_pool)
        print(name, "test", len(test), "pool", len(pool), "dev", len(dev), flush=True)
        summary, detail = evaluate_block(
            models, test, pool, methods, settings, cfg, n_agents=2, mode="random", seed=0, horizons=HORIZONS, label=name
        )
        output["datasets"][name] = summary
        details[name] = detail
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Bootstrap on Hotpot mixed, TrustMem minus CRAG and CRAG+, question level, one plant.
    output["bootstrap"] = {}
    hot = details["hotpot"]["mixed"]
    for other in ("crag", "crag_plus", "marag"):
        output["bootstrap"][f"f1_vs_{other}"] = bootstrap_delta(hot["trustmem"]["f1s"], hot[other]["f1s"])
        if "1000" in hot["trustmem"]["delayed_f1s"]:
            output["bootstrap"][f"H1000_vs_{other}"] = bootstrap_delta(
                hot["trustmem"]["delayed_f1s"]["1000"], hot[other]["delayed_f1s"]["1000"]
            )
    print("bootstrap", output["bootstrap"], flush=True)

    # Non-oracle is already the main split. Assigned-support control on Hotpot mixed only.
    print("assigned-support control", flush=True)
    _, test, pool = load_slices("hotpot", 120, 200)
    assigned, _ = evaluate_block(
        models,
        test,
        pool,
        ["crag", "trustmem"],
        ["mixed"],
        cfg,
        n_agents=2,
        mode="assigned",
        seed=0,
        horizons=(4, 100),
        label="assigned",
    )
    output["assigned_mixed"] = assigned["mixed"]

    print("agent scale", flush=True)
    output["agents"] = {}
    hot_mixed = output["datasets"]["hotpot"]["mixed"]
    output["agents"]["2"] = {
        method: {
            "n": hot_mixed[method]["n"],
            "f1": hot_mixed[method]["f1"],
            "delayed": {key: hot_mixed[method]["delayed"][key] for key in ("4", "100") if key in hot_mixed[method]["delayed"]},
            "hold": {key: hot_mixed[method]["hold"][key] for key in ("4", "100") if key in hot_mixed[method]["hold"]},
        }
        for method in ("crag", "trustmem")
    }
    _, test, pool = load_slices("hotpot", 120, 200)
    for n_agents in (4, 8):
        block, _ = evaluate_block(
            models,
            test,
            pool,
            ["crag", "trustmem"],
            ["mixed"],
            cfg,
            n_agents=n_agents,
            mode="random",
            seed=0,
            horizons=(4, 100),
            label=f"agents{n_agents}",
        )
        output["agents"][str(n_agents)] = block["mixed"]
        OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("sensitivity on dev", flush=True)
    dev, _, pool = load_slices("hotpot", 120, 8)
    output["sensitivity"] = []
    for theta in (0.10, 0.20, 0.50):
        for tau in (0.30, 0.50, 0.70):
            trial = base_cfg(theta, tau)
            block, _ = evaluate_block(
                models,
                dev,
                pool,
                ["trustmem"],
                ["mixed"],
                trial,
                n_agents=2,
                mode="random",
                seed=0,
                horizons=(4,),
                label=f"sens-{theta}-{tau}",
            )
            row = {"theta_entail": theta, "semantic_tau": tau, "f1": block["mixed"]["trustmem"]["f1"], "H4": block["mixed"]["trustmem"]["delayed"]["4"]}
            output["sensitivity"].append(row)
            print(" ", row, flush=True)
            OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")

    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
