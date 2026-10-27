"""TrustMem evaluation on HotpotQA and 2WikiMultihopQA.

Specialist agents hold disjoint document sets and communicate evidence
contracts. All methods share one frozen extractive reader, so measured
gaps come from admission and repair rather than from a stronger generator.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "trustmem_results.json"
TRACE_PATH = ROOT / "results" / "example_trace.json"

STOP = set(
    "a an the of to in on for and or is was were be been being by with from as at "
    "that this it its their his her she he they them who whom which what when where "
    "how why into over under about than then so if not no yes do does did".split()
)

FAULTS = ["unsupported", "incomplete", "stale", "conflict"]
METHODS = ["nl", "debate", "central", "marag", "trustmem"]
HONEST_CACHE: dict = {}
# Four later questions compete for eight memory slots. Recency then drops
# the oldest page; strength keeps a verified page that still bridges.
DELAY = 4
CAP = 8


def is_specialist(producer: str) -> bool:
    return producer.startswith("agent_")


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = {}
    for tok in pred_toks:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in gold_toks:
        if common.get(tok, 0):
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def n_words(text: str) -> int:
    return len(text.split())


@dataclass
class Contract:
    claim: str
    evidence: list
    producer: str
    query_id: str
    confidence: float
    t_start: int
    t_end: int
    fault: str = "none"
    injected: bool = False
    strength: float = 0.0
    owned: tuple = ()

    def evidence_text(self) -> str:
        parts = []
        for ev in self.evidence:
            title = ev.get("title", "")
            parts.append(f"{title}: {ev['text']}" if title else ev["text"])
        return "\n".join(parts)

    def blob(self) -> str:
        titles = " ".join(ev.get("title", "") for ev in self.evidence)
        return f"{self.claim} {titles} {self.evidence_text()}"


@dataclass
class Models:
    device: str
    qa_tok: object
    qa: object
    nli_tok: object
    nli: object
    emb_tok: object
    emb: object
    nli_cache: dict = field(default_factory=dict)
    emb_cache: dict = field(default_factory=dict)

    def answer(self, question: str, context: str) -> tuple[str, float]:
        if not context.strip():
            return "", 0.0
        enc = self.qa_tok(
            question,
            context,
            truncation="only_second",
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        seq_ids = enc.sequence_ids(0)
        batch = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.qa(**batch)
        start_probs = torch.softmax(out.start_logits[0], dim=-1).detach().cpu().numpy()
        end_probs = torch.softmax(out.end_logits[0], dim=-1).detach().cpu().numpy()
        best = None
        limit = len(seq_ids)
        for i in range(limit):
            if seq_ids[i] != 1:
                continue
            for j in range(i, min(i + 24, limit)):
                if seq_ids[j] != 1:
                    break
                score = float(start_probs[i] * end_probs[j])
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best is None:
            return "", 0.0
        char_s = offsets[best[1]][0]
        char_e = offsets[best[2]][1]
        if char_s is None or char_e is None or char_e <= char_s:
            return "", 0.0
        span = context[char_s:char_e].strip()
        return span, best[0]

    def nli_probs(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        premise = premise.strip()[:1200]
        hypothesis = hypothesis.strip()[:300]
        if not premise or not hypothesis:
            return (0.0, 0.0, 1.0)
        key = (premise, hypothesis)
        if key in self.nli_cache:
            return self.nli_cache[key]
        enc = self.nli_tok(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.nli(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        packed = (float(probs[0]), float(probs[1]), float(probs[2]))
        self.nli_cache[key] = packed
        return packed

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        return self.nli_probs(premise, hypothesis)[1]

    def contradict_prob(self, premise: str, hypothesis: str) -> float:
        return self.nli_probs(premise, hypothesis)[0]

    def embed(self, text: str, query: bool) -> np.ndarray:
        raw = text.strip()[:800]
        key = (query, raw)
        if key in self.emb_cache:
            return self.emb_cache[key]
        payload = raw
        if query:
            payload = "Represent this sentence for searching relevant passages: " + raw
        enc = self.emb_tok(
            payload,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = self.emb(**enc).last_hidden_state[0, 0]
        vec = hidden.detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm
        self.emb_cache[key] = vec
        return vec

    def cosine(self, question: str, evidence: str) -> float:
        if not evidence.strip():
            return 0.0
        a = self.embed(question, query=True)
        b = self.embed(evidence, query=False)
        return float(np.dot(a, b))


def load_models(device: str) -> Models:
    qa_name = "deepset/deberta-v3-base-squad2"
    nli_name = "cross-encoder/nli-deberta-v3-base"
    emb_name = "BAAI/bge-small-en-v1.5"
    qa_tok = AutoTokenizer.from_pretrained(qa_name)
    qa = AutoModelForQuestionAnswering.from_pretrained(qa_name).to(device).eval()
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device).eval()
    emb_tok = AutoTokenizer.from_pretrained(emb_name)
    emb = AutoModel.from_pretrained(emb_name).to(device).eval()
    return Models(device, qa_tok, qa, nli_tok, nli, emb_tok, emb)


def parse_example(ex) -> dict:
    titles = list(ex["context"]["title"])
    sent_groups = list(ex["context"]["sentences"])
    support_titles = set(ex["supporting_facts"]["title"])
    docs = []
    for i, (title, sentences) in enumerate(zip(titles, sent_groups)):
        clean = []
        for sent in sentences:
            text = re.sub(r"\s+", " ", str(sent)).strip()
            if text:
                clean.append(text)
        docs.append(
            {
                "doc_id": f"{i}:{title}",
                "title": title,
                "sentences": clean,
                "support": title in support_titles,
            }
        )
    return {
        "id": str(ex["id"]),
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type") or "unknown",
        "docs": docs,
    }


def answer_supported(ex: dict) -> bool:
    blob = normalize_answer(" ".join(sent for doc in ex["docs"] for sent in doc["sentences"]))
    ans = normalize_answer(ex["answer"])
    return bool(ans) and ans in blob and len(ans.split()) <= 12


def load_examples(dataset: str, n_dev: int, n_test: int, seed: int) -> tuple[list, list]:
    if dataset == "hotpot":
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
    elif dataset == "2wiki":
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    else:
        raise ValueError(dataset)
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
    dev = parsed[:n_dev]
    test = parsed[n_dev : n_dev + n_test]
    return dev, test


def bm25_sentences(question: str, docs: list, k: int) -> list:
    rows = []
    for doc in docs:
        for sid, text in enumerate(doc["sentences"]):
            rows.append((doc, sid, text))
    if not rows:
        return []
    tokenized = [re.findall(r"[a-z0-9]+", text.lower()) for _, _, text in rows]
    ranker = BM25Okapi(tokenized)
    query = re.findall(r"[a-z0-9]+", question.lower())
    scores = ranker.get_scores(query if query else ["unknown"])
    order = np.argsort(-scores)
    picked = []
    for idx in order[:k]:
        doc, sid, text = rows[int(idx)]
        picked.append(
            {
                "doc_id": doc["doc_id"],
                "sent_id": sid,
                "title": doc["title"],
                "text": text,
            }
        )
    return picked


def split_docs(docs: list) -> tuple[list, list]:
    support = [d for d in docs if d["support"]]
    other = [d for d in docs if not d["support"]]
    a, c = [], []
    for i, doc in enumerate(support):
        (a if i % 2 == 0 else c).append(doc)
    if support and not c:
        # Keep at least one supporting page on each side when possible.
        if len(a) > 1:
            c.append(a.pop())
    for i, doc in enumerate(other):
        (a if i % 2 == 0 else c).append(doc)
    if not a and c:
        a.append(c.pop())
    if not c and a:
        c.append(a.pop())
    return a, c


def doc_index(docs: list) -> dict:
    return {d["doc_id"]: d for d in docs}


def top_paragraphs(question: str, docs: list, k: int) -> list:
    usable = [doc for doc in docs if doc["sentences"]]
    if not usable:
        return []
    tokenized = [re.findall(r"[a-z0-9]+", " ".join(doc["sentences"]).lower()) or ["empty"] for doc in usable]
    ranker = BM25Okapi(tokenized)
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    scores = ranker.get_scores(query)
    order = np.argsort(-np.asarray(scores))[:k]
    return [usable[int(idx)] for idx in order]


def make_honest_contract(models: Models, question: str, docs: list, producer: str, qid: str, t: int) -> Contract:
    evidence = bm25_sentences(question, top_paragraphs(question, docs, k=1), k=4)
    context = "\n".join(f"{ev['title']}: {ev['text']}" for ev in evidence)
    claim, score = models.answer(question, context)
    return Contract(
        claim=claim,
        evidence=evidence,
        producer=producer,
        query_id=qid,
        confidence=float(min(score, 0.95)),
        t_start=t,
        t_end=t + 20,
        fault="none",
        injected=False,
        owned=tuple(doc["doc_id"] for doc in docs),
    )


def salient_span(text: str, banned: set[str]) -> str:
    pieces = re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b", text)
    pieces = [p for p in pieces if normalize_answer(p) not in banned and len(p) > 2]
    if pieces:
        pieces.sort(key=len, reverse=True)
        return pieces[0]
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text)
    for word in words:
        if normalize_answer(word) not in banned:
            return word
    return "unknown"


def prov_ok(contract: Contract, docs: dict) -> bool:
    if not contract.evidence:
        return False
    for ev in contract.evidence:
        doc = docs.get(ev["doc_id"])
        if doc is None:
            return False
        if ev["text"] not in doc["sentences"]:
            return False
    return True


def semantic_prov(models: Models, contract: Contract, docs: dict, tau: float) -> bool:
    """A non-verbatim quote stands only when the page entails it and states the claim."""
    if not contract.evidence:
        return False
    for ev in contract.evidence:
        doc = docs.get(ev["doc_id"])
        if doc is None or not ev.get("text"):
            return False
        page = " ".join(doc["sentences"])[:1200]
        if models.entail_prob(page, ev["text"]) < tau:
            return False
        if contract.claim and not grounded(contract.claim, page):
            return False
    return True


def grounded(claim: str, evidence_text: str) -> bool:
    if not claim or not evidence_text:
        return False
    if normalize_answer(claim) in {"yes", "no"}:
        return True
    return normalize_answer(claim) in normalize_answer(evidence_text)


def internal_support(models: Models, contract: Contract, cfg: dict) -> tuple[bool, float]:
    text = contract.evidence_text()
    if not grounded(contract.claim, text):
        return False, 0.0
    if not cfg["entail"]:
        return True, 1.0
    hyp = f"The text states {contract.claim}."
    prob = models.entail_prob(text, hyp)
    return prob >= cfg["theta_entail"], prob


def relevant(models: Models, contract: Contract, question: str, cfg: dict) -> tuple[bool, float]:
    q_toks = tokens(question)
    b_toks = tokens(contract.blob())
    overlap = (len(q_toks & b_toks) / len(q_toks)) if q_toks else 0.0
    cos = models.cosine(question, contract.evidence_text())
    score = 0.45 * overlap + 0.55 * max(cos, 0.0)
    return score >= cfg["theta_app"], score


def qscore(models: Models, contract: Contract, question: str) -> float:
    hyp = f"Question: {question} Answer: {contract.claim}."
    return models.entail_prob(contract.evidence_text(), hyp)


def expand_contract(contract: Contract, docs: dict, question: str) -> Contract:
    cited = []
    seen = {(ev["doc_id"], ev["sent_id"]) for ev in contract.evidence}
    doc_ids = []
    for ev in contract.evidence:
        if ev["doc_id"] not in doc_ids:
            doc_ids.append(ev["doc_id"])
    pool_docs = [docs[did] for did in doc_ids if did in docs]
    extra = bm25_sentences(question, pool_docs, k=4)
    evidence = list(contract.evidence)
    for ev in extra:
        key = (ev["doc_id"], ev["sent_id"])
        if key in seen:
            continue
        evidence.append(ev)
        seen.add(key)
        if len(evidence) >= 5:
            break
    return Contract(
        claim=contract.claim,
        evidence=evidence,
        producer=contract.producer,
        query_id=contract.query_id,
        confidence=contract.confidence,
        t_start=contract.t_start,
        t_end=contract.t_end,
        fault=contract.fault,
        injected=contract.injected,
    )


def locate_answer(docs: list, answer: str):
    target = normalize_answer(answer)
    for doc in docs:
        for sid, text in enumerate(doc["sentences"]):
            if target and target in normalize_answer(text):
                return doc, sid, text
    for doc in docs:
        if doc["sentences"]:
            return doc, 0, doc["sentences"][0]
    return None, None, ""


def false_entity(docs: list, banned: set[str], rng: random.Random) -> str:
    pool = [doc for doc in docs if not doc.get("support") and doc["sentences"]]
    rng.shuffle(pool)
    for doc in pool:
        span = salient_span(" ".join(doc["sentences"][:3]), banned)
        if span and normalize_answer(span) not in banned and span.lower() != "unknown":
            return span
    return ""


def edited_sentence(text: str, answer: str, false_claim: str) -> str:
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    if answer and pattern.search(text):
        return pattern.sub(false_claim, text, count=1)
    return f"{false_claim} is recorded in this source. {text}"


def build_fault_contract(
    models: Models,
    question: str,
    docs: list,
    honest: list[Contract],
    fault: str,
    qid: str,
    t: int,
    rng: random.Random,
    answer: str,
) -> tuple[Contract | None, list]:
    del models
    host, sid, text = locate_answer(docs, answer)
    if host is None:
        return None, []
    banned = {normalize_answer(answer)}
    banned.update(normalize_answer(c.claim) for c in honest if c.claim)
    planted = false_entity(docs, banned, rng)
    if not planted:
        return None, []
    holder = honest[0].producer
    for contract in honest:
        if any(ev["doc_id"] == host["doc_id"] for ev in contract.evidence):
            holder = contract.producer
            break
    edited = edited_sentence(text, answer, planted)

    owned_by = {contract.producer: tuple(contract.owned) for contract in honest}

    if fault == "unsupported":
        fake = {"doc_id": host["doc_id"], "sent_id": -1, "title": host["title"], "text": edited}
        contract = Contract(planted, [fake], holder, qid, 1.0, t, t + 20, fault, True, owned=owned_by.get(holder, ()))
        return contract, []

    if fault == "incomplete":
        kept = []
        for sent_id, sent in enumerate(host["sentences"]):
            if normalize_answer(answer) in normalize_answer(sent):
                continue
            kept.append({"doc_id": host["doc_id"], "sent_id": sent_id, "title": host["title"], "text": sent})
            break
        if not kept:
            return None, []
        claim = next((c.claim for c in honest if c.producer == holder and c.claim), planted)
        contract = Contract(claim, kept, holder, qid, 0.4, t, t + 20, fault, False, owned=owned_by.get(holder, ()))
        return contract, []

    if fault == "stale":
        archive = {
            "doc_id": f"archive:{host['doc_id']}",
            "title": host["title"],
            "sentences": [edited],
            "support": False,
        }
        ev = {"doc_id": archive["doc_id"], "sent_id": 0, "title": archive["title"], "text": edited}
        contract = Contract(planted, [ev], "interferer", qid, 1.0, 0, t - 1, fault, True, owned=(archive["doc_id"],))
        return contract, [archive]

    if fault == "paraphrase":
        claim = next((c.claim for c in honest if c.producer == holder and c.claim), answer)
        text = f"A restatement records {claim} as the fact on this page."
        fake = {"doc_id": host["doc_id"], "sent_id": -1, "title": host["title"], "text": text}
        contract = Contract(
            claim, [fake], holder, qid, 0.9, t, t + 20, fault, True, owned=owned_by.get(holder, ())
        )
        return contract, []

    if fault == "conflict":
        distractors = [doc for doc in docs if not doc["support"] and doc["sentences"]]
        if not distractors:
            return None, []
        q_toks = tokens(question)
        ranked = []
        for doc in distractors:
            for sent_id, sent in enumerate(doc["sentences"]):
                ranked.append((len(q_toks & tokens(sent)), doc, sent_id, sent))
        ranked.sort(key=lambda row: row[0], reverse=True)
        _, doc, sent_id, sent = ranked[0]
        claim = salient_span(sent, banned) or planted
        ev = {"doc_id": doc["doc_id"], "sent_id": sent_id, "title": doc["title"], "text": sent}
        return Contract(claim, [ev], "interferer", qid, 1.0, t, t + 20, fault, True, owned=(doc["doc_id"],)), []
    raise ValueError(fault)


def apply_fault(honest: list[Contract], fault_contract: Contract | None, fault: str) -> list[Contract]:
    peers = [honest[0], honest[1]]
    if fault == "none" or fault_contract is None:
        return peers
    if fault in {"incomplete", "unsupported"}:
        replaced = False
        out = []
        for peer in peers:
            if not replaced and peer.producer == fault_contract.producer:
                out.append(fault_contract)
                replaced = True
            else:
                out.append(peer)
        return out
    return peers + [fault_contract]


def cited_docs(contract: Contract, docs: dict) -> list:
    pool = []
    seen = set()
    for ev in contract.evidence:
        doc = docs.get(ev["doc_id"])
        if doc is not None and ev["doc_id"] not in seen:
            pool.append(doc)
            seen.add(ev["doc_id"])
    return pool


def refetch_contract(models: Models, contract: Contract, docs: dict, question: str) -> Contract:
    pool = cited_docs(contract, docs)
    evidence = bm25_sentences(question, pool, k=4)
    q_toks = tokens(question)
    best = None
    best_score = -1
    for doc in pool:
        for sent_id, sent in enumerate(doc["sentences"]):
            score = len(q_toks & tokens(f"{doc['title']} {sent}"))
            if sent and score > best_score:
                best_score = score
                best = {"doc_id": doc["doc_id"], "sent_id": sent_id, "title": doc["title"], "text": sent}
    if best is not None and all(item.get("text") != best["text"] for item in evidence):
        evidence = [best] + evidence
    if not evidence:
        return contract
    restored = " ".join(ev["text"] for ev in evidence)
    context = "\n".join(f"{ev['title']}: {ev['text']}" for ev in evidence)
    reread, score = models.answer(question, context)
    if grounded(contract.claim, restored):
        claim, injected = contract.claim, contract.injected
    else:
        claim, injected = reread, False
    return Contract(
        claim,
        evidence,
        contract.producer,
        contract.query_id,
        float(max(contract.confidence, score)),
        contract.t_start,
        contract.t_end,
        contract.fault,
        injected,
        owned=tuple(contract.owned),
    )


def admit_trustmem(
    models: Models,
    peers: list[Contract],
    question: str,
    t: int,
    docs: dict,
    cfg: dict,
    allow_expand: bool,
) -> tuple[list[Contract], int]:
    checks = 0
    kept = []
    for contract in peers:
        if cfg["temp"] and not (contract.t_start <= t <= contract.t_end):
            continue
        quote_ok = (not cfg["prov"]) or prov_ok(contract, docs)
        if (not quote_ok) and cfg.get("semantic") and cited_docs(contract, docs):
            quote_ok = semantic_prov(models, contract, docs, float(cfg.get("semantic_tau", 0.5)))
        if (not quote_ok) and cfg["request"] and allow_expand and cited_docs(contract, docs):
            contract = refetch_contract(models, contract, docs, question)
            checks += 1
            quote_ok = prov_ok(contract, docs)
        if cfg["prov"] and not quote_ok:
            continue
        if (
            cfg["request"]
            and allow_expand
            and contract.claim
            and not grounded(contract.claim, contract.evidence_text())
            and cited_docs(contract, docs)
        ):
            contract = refetch_contract(models, contract, docs, question)
            checks += 1
        if cfg["entail"] and contract.claim:
            ok, _ = internal_support(models, contract, cfg)
            checks += 1
            if not ok:
                contract = Contract(
                    "",
                    contract.evidence,
                    contract.producer,
                    contract.query_id,
                    contract.confidence,
                    contract.t_start,
                    contract.t_end,
                    contract.fault,
                    False,
                )
        rel_ok, _ = relevant(models, contract, question, cfg)
        if not rel_ok or not contract.evidence:
            continue
        kept.append(contract)
    if cfg["qscore"] and kept:
        claimed = [contract for contract in kept if contract.claim]
        rest = [contract for contract in kept if not contract.claim]
        if claimed:
            scored = []
            for contract in claimed:
                score = qscore(models, contract, question)
                checks += 1
                scored.append((score, contract))
            best = max(score for score, _ in scored)
            floor = best - cfg["margin"]
            if best >= cfg["theta_q"]:
                floor = max(cfg["theta_q"], floor)
            claimed = [contract for score, contract in scored if score >= floor - 1e-8]
        kept = rest + claimed
    return kept, checks


def admit_central(models: Models, peers: list[Contract], cfg: dict) -> tuple[list[Contract], int]:
    kept = []
    checks = 0
    for contract in peers:
        ok, _ = internal_support(models, contract, cfg)
        checks += 1 if cfg["entail"] else 0
        if ok:
            kept.append(contract)
    return kept, checks


def reader_from_contracts(models: Models, question: str, contracts: list[Contract]) -> tuple[str, float, int]:
    q_toks = tokens(question)
    pieces = []
    for contract in contracts:
        if contract.evidence:
            for ev in contract.evidence:
                text = f"{ev.get('title', '')}: {ev['text']}".strip()
                overlap = len(q_toks & tokens(text))
                pieces.append((overlap, text))
        elif contract.claim:
            pieces.append((0, f"The answer is {contract.claim}."))
    pieces.sort(key=lambda item: item[0], reverse=True)
    kept = []
    used = 0
    seen = set()
    for _, text in pieces:
        if text in seen:
            continue
        seen.add(text)
        words = n_words(text)
        if kept and used + words > 360:
            break
        kept.append(text)
        used += words
    pred, score = models.answer(question, "\n".join(kept))
    return pred, score, used


def clone_contract(contract: Contract, evidence: list | None = None, claim: str | None = None) -> Contract:
    return Contract(
        claim if claim is not None else contract.claim,
        list(contract.evidence if evidence is None else evidence),
        contract.producer,
        contract.query_id,
        contract.confidence,
        contract.t_start,
        contract.t_end,
        contract.fault,
        contract.injected,
        contract.strength,
        tuple(contract.owned),
    )


def bridge_tokens(contracts: list[Contract]) -> set[str]:
    by_producer: dict[str, list[str]] = defaultdict(list)
    for contract in contracts:
        by_producer[contract.producer].append(contract.evidence_text())
    sets = [tokens(" ".join(parts)) for parts in by_producer.values() if any(parts)]
    shared: set[str] = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared |= sets[i] & sets[j]
    return shared


def merge_replies(admitted: list[Contract], replies: list[Contract]) -> list[Contract]:
    """Attach attested sentences to the specialist that returned them, without adding a memory slot."""
    for reply in replies:
        host = next(
            (
                contract
                for contract in admitted
                if contract.producer == reply.producer and contract.query_id == reply.query_id
            ),
            None,
        )
        if host is None:
            admitted.append(reply)
            continue
        existing = {(ev.get("doc_id"), ev.get("text")) for ev in host.evidence}
        for ev in reply.evidence:
            key = (ev.get("doc_id"), ev.get("text"))
            if key not in existing:
                host.evidence.append(ev)
                existing.add(key)
        if reply.owned:
            host.owned = tuple(dict.fromkeys((*host.owned, *reply.owned)))
    return admitted


def peer_attest(admitted: list[Contract], question: str, docs: dict) -> list[Contract]:
    """Each specialist searches only the pages it holds and returns sentences that name an entity already verified by the other specialist."""
    verified = [contract for contract in admitted if contract.evidence and prov_ok(contract, docs)]
    by_producer: dict[str, list[Contract]] = defaultdict(list)
    for contract in verified:
        if is_specialist(contract.producer):
            by_producer[contract.producer].append(contract)
    if len(by_producer) < 2:
        return []
    question_tokens = tokens(question)
    replies = []
    for asker, own in by_producer.items():
        wanted: set[str] = set()
        for producer, group in by_producer.items():
            if producer == asker:
                continue
            for contract in group:
                wanted |= tokens(contract.evidence_text())
        wanted -= question_tokens
        if not wanted:
            continue
        pool = []
        seen = set()
        for contract in own:
            doc_ids = contract.owned or tuple(ev["doc_id"] for ev in contract.evidence)
            for doc_id in doc_ids:
                doc = docs.get(doc_id)
                if doc is None or doc_id in seen:
                    continue
                seen.add(doc_id)
                pool.append(doc)
        hits = []
        for doc in pool:
            for sent_id, sent in enumerate(doc["sentences"]):
                overlap = tokens(f"{doc['title']} {sent}") & wanted
                if overlap:
                    hits.append((len(overlap), doc, sent_id, sent))
        hits.sort(key=lambda row: row[0], reverse=True)
        evidence = []
        used = set()
        for _, doc, sent_id, sent in hits:
            key = (doc["doc_id"], sent_id)
            if key in used:
                continue
            used.add(key)
            evidence.append({"doc_id": doc["doc_id"], "sent_id": sent_id, "title": doc["title"], "text": sent})
            if len(evidence) >= 2:
                break
        if not evidence:
            continue
        replies.append(
            Contract(
                claim="",
                evidence=evidence,
                producer=asker,
                query_id=own[0].query_id,
                confidence=0.55,
                t_start=own[0].t_start,
                t_end=own[0].t_end,
                fault="attest",
                injected=False,
                owned=tuple(doc["doc_id"] for doc in pool),
            )
        )
    return replies


def citation_hop(admitted: list[Contract], question: str, docs: dict) -> list[Contract]:
    """Open the page whose title another specialist's page actually names, and keep the sentence that names it."""
    verified = [contract for contract in admitted if contract.evidence and prov_ok(contract, docs)]
    by_producer: dict[str, list[Contract]] = defaultdict(list)
    for contract in verified:
        if is_specialist(contract.producer):
            by_producer[contract.producer].append(contract)
    if len(by_producer) < 2:
        return []
    owned_by: dict[str, list] = {}
    cited_by: dict[str, list[str]] = defaultdict(list)
    for producer, group in by_producer.items():
        pool = []
        seen = set()
        for contract in group:
            for doc_id in contract.owned or tuple(ev.get("doc_id", "") for ev in contract.evidence):
                doc = docs.get(doc_id)
                if doc is None or doc_id in seen:
                    continue
                seen.add(doc_id)
                pool.append(doc)
            for ev in contract.evidence:
                doc_id = ev.get("doc_id", "")
                if doc_id and doc_id not in cited_by[producer]:
                    cited_by[producer].append(doc_id)
        owned_by[producer] = pool
    q_toks = tokens(question)
    replies = []
    for asker, pool in owned_by.items():
        other_blob = []
        namer_docs = []
        for producer, doc_ids in cited_by.items():
            if producer == asker:
                continue
            for doc_id in doc_ids:
                doc = docs.get(doc_id)
                if doc is None:
                    continue
                other_blob.append(" ".join(doc["sentences"]))
                namer_docs.append((producer, doc))
        blob = " ".join(other_blob).lower()
        if not blob:
            continue
        already = {ev.get("doc_id") for contract in by_producer[asker] for ev in contract.evidence}
        evidence = []
        mention_by: dict[str, list] = defaultdict(list)
        for doc in pool:
            title = (doc.get("title") or "").strip()
            if len(title) <= 3 or doc["doc_id"] in already or title.lower() not in blob:
                continue
            best = None
            best_score = -1
            for sent_id, sent in enumerate(doc["sentences"]):
                score = len(q_toks & tokens(f"{doc['title']} {sent}"))
                if sent and score > best_score:
                    best_score = score
                    best = {"doc_id": doc["doc_id"], "sent_id": sent_id, "title": doc["title"], "text": sent}
            if best is not None:
                evidence.append(best)
            needle = title.lower()
            for producer, namer in namer_docs:
                for sent_id, sent in enumerate(namer["sentences"]):
                    if needle in sent.lower():
                        mention_by[producer].append(
                            {"doc_id": namer["doc_id"], "sent_id": sent_id, "title": namer["title"], "text": sent}
                        )
                        break
            if len(evidence) >= 2:
                break
        if evidence:
            host = by_producer[asker][0]
            replies.append(
                Contract(
                    claim="",
                    evidence=evidence,
                    producer=asker,
                    query_id=host.query_id,
                    confidence=0.55,
                    t_start=host.t_start,
                    t_end=host.t_end,
                    fault="attest",
                    injected=False,
                    owned=tuple(doc["doc_id"] for doc in pool),
                )
            )
        for producer, mentions in mention_by.items():
            host = by_producer[producer][0]
            replies.append(
                Contract(
                    claim="",
                    evidence=mentions[:2],
                    producer=producer,
                    query_id=host.query_id,
                    confidence=0.55,
                    t_start=host.t_start,
                    t_end=host.t_end,
                    fault="attest",
                    injected=False,
                    owned=host.owned,
                )
            )
    return replies


def citation_prune(admitted: list[Contract], docs: dict) -> tuple[list[Contract], bool]:
    """Once a specialist holds a page that another page names, drop that specialist's uncited distractors."""
    pages: dict[str, dict] = {}
    for contract in admitted:
        for ev in contract.evidence:
            doc_id = ev.get("doc_id") or ev.get("text", "")[:40]
            slot = pages.setdefault(
                doc_id,
                {"producer": contract.producer, "title": ev.get("title") or "", "hits": 0},
            )
            slot["hits"] += 1
            if ev.get("title"):
                slot["title"] = ev["title"]
    if len(pages) < 2:
        return admitted, False

    def body(doc_id: str) -> str:
        doc = docs.get(doc_id)
        if doc is not None:
            return " ".join(doc["sentences"]).lower()
        return ""

    linked = {doc_id: False for doc_id in pages}
    ids = list(pages)
    for left_i, left in enumerate(ids):
        for right in ids[left_i + 1 :]:
            title_l = pages[left]["title"].strip()
            title_r = pages[right]["title"].strip()
            hit = (len(title_l) > 3 and title_l.lower() in body(right)) or (
                len(title_r) > 3 and title_r.lower() in body(left)
            )
            if hit:
                linked[left] = True
                linked[right] = True
    if not any(linked.values()):
        return admitted, False
    producers_with_link = {pages[doc_id]["producer"] for doc_id, ok in linked.items() if ok}
    changed = False
    rebuilt = []
    for contract in admitted:
        if contract.producer not in producers_with_link:
            rebuilt.append(contract)
            continue
        evidence = [ev for ev in contract.evidence if linked.get(ev.get("doc_id") or ev.get("text", "")[:40], False)]
        if len(evidence) != len(contract.evidence):
            changed = True
        if evidence:
            rebuilt.append(clone_contract(contract, evidence=evidence))
        else:
            changed = True
    return (rebuilt or admitted), changed


def close_contracts(
    models: Models, contracts: list[Contract], question: str, cfg: dict, limit: int = 4
) -> tuple[list[Contract], int]:
    """Show the reader four sentences. Attestation keeps a sentence only when it names an entity shared by the two specialists."""
    del models
    usable = [contract for contract in contracts if contract.evidence or contract.claim]
    if not usable:
        return [], 0
    rows = []
    for contract in usable:
        evidence = contract.evidence or [
            {"doc_id": "", "sent_id": 0, "title": "message", "text": contract.claim}
        ]
        for ev in evidence:
            rows.append((contract, ev))
    if not rows:
        return [], 0
    if cfg.get("attest", False):
        # The bridge is the intersection of the two specialists, not a token shared with a third sender.
        by_specialist: dict[str, list[str]] = defaultdict(list)
        for contract in usable:
            if is_specialist(contract.producer) and contract.evidence_text():
                by_specialist[contract.producer].append(contract.evidence_text())
        shared: set[str] = set()
        if len(by_specialist) >= 2:
            sets = [tokens(" ".join(parts)) for parts in by_specialist.values()]
            if cfg.get("attest_pair"):
                shared = set()
                for left in range(len(sets)):
                    for right in range(left + 1, len(sets)):
                        shared |= sets[left] & sets[right]
            else:
                shared = set.intersection(*sets)
            shared -= tokens(question)
        if shared and cfg.get("attest_pair"):
            rows = [
                (contract, ev)
                for contract, ev in rows
                if is_specialist(contract.producer)
                or (tokens(f"{ev.get('title', '')} {ev.get('text', '')}") & shared)
            ] or rows
        elif shared:
            bound_producers = set()
            for contract, ev in rows:
                if not is_specialist(contract.producer):
                    continue
                if tokens(f"{ev.get('title', '')} {ev.get('text', '')}") & shared:
                    bound_producers.add(contract.producer)
            bound = []
            for contract, ev in rows:
                toks = tokens(f"{ev.get('title', '')} {ev.get('text', '')}")
                if is_specialist(contract.producer):
                    if contract.producer in bound_producers:
                        bound.append((contract, ev))
                elif toks & shared:
                    bound.append((contract, ev))
            if bound:
                rows = bound
        else:
            rows = [row for row in rows if is_specialist(row[0].producer)] or rows
    tokenized = [re.findall(r"[a-z0-9]+", ev.get("text", "").lower()) or ["empty"] for _, ev in rows]
    ranker = BM25Okapi(tokenized)
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    ranked = []
    for (contract, ev), bm25_score in zip(rows, ranker.get_scores(query)):
        ranked.append((float(bm25_score), contract, ev))
    ranked.sort(key=lambda row: row[0], reverse=True)
    picked = []
    seen_producers = set()
    for score, contract, ev in ranked:
        if contract.producer in seen_producers:
            continue
        picked.append((score, contract, ev))
        seen_producers.add(contract.producer)
    for row in ranked:
        if row not in picked:
            picked.append(row)
        if len(picked) >= limit:
            break
    grouped: dict[int, list] = defaultdict(list)
    parents: dict[int, Contract] = {}
    for _, contract, ev in picked[:limit]:
        grouped[id(contract)].append(ev)
        parents[id(contract)] = contract
    return [clone_contract(parents[key], evidence=evs) for key, evs in grouped.items()], 0


def crag_select(
    peers: list[Contract],
    question: str,
    docs: dict,
    qid: str,
    allow_expand: bool,
) -> tuple[list[Contract], int]:
    """Corrective RAG: a quote that is not on its cited page is replaced by one retrieval from that page. Page-open rights match TrustMem."""
    fresh = [contract for contract in peers if contract.query_id == qid] or list(peers)
    collected = []
    checks = 0
    for contract in fresh:
        on_page = bool(contract.evidence) and prov_ok(contract, docs)
        if allow_expand and contract.evidence and not on_page:
            retrieved = bm25_sentences(question, cited_docs(contract, docs), k=2)
            checks += 1
            if retrieved:
                collected.extend(retrieved)
                continue
        if contract.evidence:
            collected.extend(contract.evidence[:2])
        elif contract.claim:
            collected.append({"doc_id": "", "sent_id": 0, "title": "message", "text": contract.claim})
    if not collected:
        return [], checks
    pseudo = [
        {
            "doc_id": f"crag:{index}",
            "title": ev.get("title", ""),
            "sentences": [ev["text"]],
            "support": False,
        }
        for index, ev in enumerate(collected)
        if ev.get("text")
    ]
    evidence = bm25_sentences(question, pseudo, k=4)
    # Restore the original page identifiers so a later verbatim check still refers to the source page.
    restored = []
    for ev in evidence:
        text = ev["text"]
        source = next((item for item in collected if item.get("text") == text), ev)
        restored.append(
            {
                "doc_id": source.get("doc_id", ""),
                "sent_id": source.get("sent_id", 0),
                "title": source.get("title", ""),
                "text": text,
            }
        )
    if not restored:
        return [], checks
    window = fresh[0]
    return [
        Contract(
            claim="",
            evidence=restored,
            producer="crag",
            query_id=qid,
            confidence=0.55,
            t_start=window.t_start,
            t_end=window.t_end,
            fault="none",
            injected=False,
            strength=0.55,
        )
    ], checks


def debate_select(models: Models, peers: list[Contract], question: str, qid: str) -> tuple[list[Contract], int]:
    fresh = [contract for contract in peers if contract.query_id == qid] or list(peers)
    best: dict[str, Contract] = {}
    for contract in fresh:
        prev = best.get(contract.producer)
        if prev is None or (contract.confidence, len(contract.evidence)) > (prev.confidence, len(prev.evidence)):
            best[contract.producer] = contract
    chosen = list(best.values())
    if len(chosen) < 2:
        return chosen, 0
    checks = 0
    kept = []
    for contract in chosen:
        self_support = 0.0
        if contract.claim and contract.evidence_text():
            self_support = models.entail_prob(contract.evidence_text(), f"The answer is {contract.claim}.")
            checks += 1
        contradicted = False
        # A page that states its own claim is kept. Multi-hop evidence from the
        # other specialist is not treated as a contradiction of that claim.
        if contract.claim and self_support < 0.35:
            for other in chosen:
                if other is contract or not other.evidence_text() or not other.claim:
                    continue
                if normalize_answer(other.claim) == normalize_answer(contract.claim):
                    continue
                supports_other = models.entail_prob(other.evidence_text(), f"The answer is {other.claim}.")
                con = models.contradict_prob(other.evidence_text(), f"The answer is {contract.claim}.")
                checks += 1
                if supports_other >= 0.35 and con >= 0.50:
                    contradicted = True
                    break
        if not contradicted:
            kept.append(contract)
    return (kept or chosen), checks


def marag_select(question: str, peers: list[Contract], qid: str) -> list[Contract]:
    pseudo = []
    for index, contract in enumerate(peers):
        sentences = [ev["text"] for ev in contract.evidence if ev.get("text")]
        if not sentences and contract.claim:
            sentences = [contract.claim]
        if not sentences:
            continue
        pseudo.append(
            {
                "doc_id": f"pool:{index}:{contract.query_id}",
                "title": contract.producer,
                "sentences": sentences,
                "support": False,
            }
        )
    evidence = bm25_sentences(question, pseudo, k=4)
    if not evidence:
        return []
    return [
        Contract(
            claim="",
            evidence=evidence,
            producer="pool",
            query_id=qid,
            confidence=0.5,
            t_start=0,
            t_end=10**9,
            fault="none",
            injected=False,
            strength=0.5,
        )
    ]


def nl_contract(peers: list[Contract], question: str, models: Models, qid: str) -> tuple[str, float, int, list[Contract]]:
    fresh = [contract for contract in peers if contract.query_id == qid] or list(peers)
    fresh = sorted(fresh, key=lambda c: c.confidence, reverse=True)
    evidence = []
    for index, contract in enumerate(fresh):
        source = contract.evidence[0]["text"] if contract.evidence else contract.claim
        if not source:
            continue
        evidence.append({"doc_id": f"nl:{index}", "sent_id": 0, "title": "message", "text": source})
        if len(evidence) >= 4:
            break
    if not evidence:
        return "", 0.0, 0, []
    packed = Contract(
        claim=fresh[0].claim if fresh else "",
        evidence=evidence,
        producer="nl",
        query_id=qid,
        confidence=float(fresh[0].confidence if fresh else 0.0),
        t_start=fresh[0].t_start if fresh else 0,
        t_end=fresh[0].t_end if fresh else 0,
        fault="none",
        injected=False,
        strength=float(fresh[0].confidence if fresh else 0.0),
    )
    pred, conf, used = reader_from_contracts(models, question, [packed])
    packed.claim = pred or packed.claim
    packed.strength = float(conf)
    return pred, conf, used, [packed]


def assign_strength(models: Models, contracts: list[Contract], question: str, t: int, docs: dict) -> None:
    for contract in contracts:
        if contract.claim and contract.evidence_text():
            support = qscore(models, contract, question)
        else:
            support = 0.45
        verified = 1.0 if prov_ok(contract, docs) else 0.35
        in_window = 1.0 if contract.t_start <= t <= contract.t_end else 0.2
        contract.strength = verified * in_window * (0.45 + 0.55 * support)


def gold_recall(text: str, gold: str) -> float:
    wanted = normalize_answer(gold).split()
    if not wanted:
        return 0.0
    have = set(normalize_answer(text).split())
    return float(sum(token in have for token in wanted) / len(wanted))


def answer_doc_ids(docs: list, answer: str) -> set[str]:
    target = normalize_answer(answer)
    found = set()
    for doc in docs:
        blob = normalize_answer(" ".join(doc["sentences"]))
        if target and target in blob:
            found.add(doc["doc_id"])
    return found


def without_answer_page(peers: list[Contract], host_ids: set[str]) -> list[Contract]:
    visible = []
    for contract in peers:
        evidence = [ev for ev in contract.evidence if ev.get("doc_id") not in host_ids]
        if contract.evidence and not evidence:
            continue
        if evidence == contract.evidence:
            visible.append(contract)
        else:
            visible.append(clone_contract(contract, evidence=evidence))
    return visible


def run_consumer(
    method: str,
    models: Models,
    question: str,
    peers: list[Contract],
    t: int,
    docs: dict,
    cfg: dict,
    gates: dict,
    allow_expand: bool,
    qid: str,
) -> dict:
    local_cfg = dict(cfg)
    local_cfg.update(gates)
    checks = 0
    if method == "nl":
        if not peers:
            pred, conf, used, admitted = "", 0.0, 0, []
        else:
            pred, conf, used, admitted = nl_contract(peers, question, models, qid)
    elif method == "debate":
        admitted, checks = debate_select(models, peers, question, qid)
        pred, conf, used = reader_from_contracts(models, question, admitted)
    elif method == "central":
        admitted, checks = admit_central(models, peers, local_cfg)
        pred, conf, used = reader_from_contracts(models, question, admitted)
    elif method == "marag":
        admitted = marag_select(question, peers, qid)
        pred, conf, used = reader_from_contracts(models, question, admitted)
    elif method == "crag":
        admitted, checks = crag_select(peers, question, docs, qid, allow_expand)
        pred, conf, used = reader_from_contracts(models, question, admitted)
    elif method == "trustmem":
        admitted, checks = admit_trustmem(models, peers, question, t, docs, local_cfg, allow_expand)
        if local_cfg.get("attest", False) and allow_expand:
            replies = peer_attest(admitted, question, docs)
            checks += len(replies)
            admitted = merge_replies(admitted, replies)
        assign_strength(models, admitted, question, t, docs)
        closed, extra = close_contracts(models, admitted, question, local_cfg)
        checks += extra
        pred, conf, used = reader_from_contracts(models, question, closed)
    else:
        raise ValueError(method)
    transmitted = sum(n_words(c.evidence_text() or c.claim) for c in peers)
    return {
        "pred": pred,
        "conf": float(conf),
        "checks": checks,
        "used_tokens": used,
        "transmitted_tokens": transmitted,
        "admitted": admitted,
    }


def memory_to_peers(memory: list[Contract]) -> list[Contract]:
    return list(memory)


def update_memory(
    memory: list[Contract],
    admitted: list[Contract],
    cap: int = CAP,
    mode: str = "recent",
) -> list[Contract]:
    merged = list(memory) + list(admitted)
    if len(merged) <= cap:
        return merged
    if mode == "recent":
        return merged[-cap:]
    order = sorted(range(len(merged)), key=lambda i: (merged[i].strength, i))
    keep = set(order[-cap:])
    return [contract for i, contract in enumerate(merged) if i in keep]


def poison_set(memory: list[Contract]) -> set[str]:
    out = set()
    for contract in memory:
        if contract.injected and contract.claim:
            out.add(normalize_answer(contract.claim))
    return out


def evaluate_split(
    models: Models,
    examples: list,
    cfg: dict,
    seeds: list[int],
    settings: list[str],
    methods: list[str],
    gates: dict,
    trace_bucket: list | None = None,
    cap: int = CAP,
    delay: int = DELAY,
) -> dict:
    per_seed = {}
    for seed in seeds:
        rng = random.Random(seed)
        buckets = {setting: {method: [] for method in methods} for setting in settings}
        memories = {setting: {method: [] for method in methods} for setting in settings}
        writes = {setting: {method: [] for method in methods} for setting in settings}
        meta = {setting: [] for setting in settings}
        for index, ex in enumerate(examples):
            qid = ex["id"]
            if qid not in HONEST_CACHE:
                a_docs, c_docs = split_docs(ex["docs"])
                t0 = 10 + index
                HONEST_CACHE[qid] = (
                    make_honest_contract(models, ex["question"], a_docs, "agent_a", qid, t0),
                    make_honest_contract(models, ex["question"], c_docs, "agent_c", qid, t0),
                )
            honest = HONEST_CACHE[qid]
            t = 10 + index
            for setting in settings:
                fault = "none" if setting == "clean" else setting
                if setting == "mixed":
                    fault = FAULTS[index % 4]
                fault_contract = None
                extra_docs = []
                if fault != "none":
                    fault_contract, extra_docs = build_fault_contract(
                        models, ex["question"], ex["docs"], list(honest), fault, qid, t, rng, ex["answer"]
                    )
                docs = doc_index(ex["docs"] + extra_docs)
                base_peers = apply_fault(list(honest), fault_contract, fault)
                injected_claim = ""
                if fault_contract is not None and fault_contract.injected:
                    injected_claim = fault_contract.claim
                meta[setting].append(
                    {
                        "question": ex["question"],
                        "answer": ex["answer"],
                        "type": ex["type"],
                        "t": t,
                        "docs": docs,
                        "base_peers": base_peers,
                        "host_ids": answer_doc_ids(ex["docs"], ex["answer"]),
                        "injected_claim": injected_claim,
                        "qid": qid,
                        "fault": fault,
                    }
                )
                for method in methods:
                    mem = memories[setting][method] if gates.get("memory", True) else []
                    peers = memory_to_peers(mem) + base_peers
                    immediate = run_consumer(
                        method, models, ex["question"], peers, t, docs, cfg, gates, allow_expand=True, qid=qid
                    )
                    new_writes = [c for c in immediate["admitted"] if c.query_id == qid]
                    mode = "strength" if method == "trustmem" and gates.get("retain", True) else "recent"
                    memories[setting][method] = update_memory(mem, new_writes, cap, mode)
                    writes[setting][method].append(new_writes)
                    pred_n = normalize_answer(immediate["pred"])
                    past_poison = poison_set(mem)
                    carry = float(pred_n in past_poison and pred_n != normalize_answer(ex["answer"])) if past_poison else 0.0
                    prop_defined = injected_claim != ""
                    prop = exact_match(immediate["pred"], injected_claim) if prop_defined else 0.0
                    admitted_hit = 0.0
                    if prop_defined:
                        admitted_hit = float(
                            any(exact_match(c.claim, injected_claim) for c in immediate["admitted"])
                        )
                    buckets[setting][method].append(
                        {
                            "em": exact_match(immediate["pred"], ex["answer"]),
                            "f1": token_f1(immediate["pred"], ex["answer"]),
                            "probe_em": 0.0,
                            "probe_f1": 0.0,
                            "hold": 0.0,
                            "prop": prop,
                            "prop_defined": prop_defined,
                            "admit": admitted_hit,
                            "admit_defined": prop_defined,
                            "carry": carry,
                            "carry_defined": bool(past_poison),
                            "conf": immediate["conf"],
                            "correct": exact_match(immediate["pred"], ex["answer"]),
                            "checks": immediate["checks"],
                            "probe_checks": 0,
                            "used_tokens": immediate["used_tokens"],
                            "transmitted_tokens": immediate["transmitted_tokens"],
                            "type": ex["type"],
                            "pred": immediate["pred"],
                        }
                    )
            if (index + 1) % 20 == 0:
                print(f"  seed {seed}: {index + 1}/{len(examples)}", flush=True)
        n = len(examples)
        for setting in settings:
            for method in methods:
                mode = "strength" if method == "trustmem" and gates.get("retain", True) else "recent"
                packets = writes[setting][method] if gates.get("memory", True) else [[] for _ in range(n)]
                for index in range(n):
                    info = meta[setting][index]
                    mem: list[Contract] = []
                    order = [index] + [(index + step) % n for step in range(1, delay + 1)]
                    for j in order:
                        mem = update_memory(mem, packets[j], cap, mode)
                    visible = without_answer_page(info["base_peers"], info["host_ids"])
                    probe = run_consumer(
                        method,
                        models,
                        info["question"],
                        mem + visible,
                        info["t"] + delay,
                        info["docs"],
                        cfg,
                        gates,
                        allow_expand=False,
                        qid=info["qid"],
                    )
                    row = buckets[setting][method][index]
                    row["probe_em"] = exact_match(probe["pred"], info["answer"])
                    row["probe_f1"] = token_f1(probe["pred"], info["answer"])
                    row["hold"] = gold_recall(" ".join(c.evidence_text() for c in mem), info["answer"])
                    row["probe_checks"] = probe["checks"]
                    if (
                        trace_bucket is not None
                        and len(trace_bucket) < 40
                        and setting == "mixed"
                        and method == "trustmem"
                        and seed == seeds[0]
                    ):
                        trace_bucket.append(
                            {
                                "id": info["qid"],
                                "dataset_question": info["question"],
                                "answer": info["answer"],
                                "fault": info["fault"],
                                "injected_claim": info["injected_claim"],
                                "pred": row["pred"],
                                "probe_pred": probe["pred"],
                                "f1": row["f1"],
                                "probe_f1": row["probe_f1"],
                                "hold": row["hold"],
                                "memory_claims": [c.claim for c in mem],
                            }
                        )
        per_seed[str(seed)] = {
            setting: {method: summarize(rows) for method, rows in methods_rows.items()}
            for setting, methods_rows in buckets.items()
        }
    return {"seeds": per_seed, "mean": mean_over_seeds(per_seed)}


def ece_score(rows: list[dict], n_bins: int = 10) -> float:
    if not rows:
        return 0.0
    confs = np.array([min(max(r["conf"], 0.0), 1.0) for r in rows], dtype=np.float64)
    correct = np.array([r["correct"] for r in rows], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (confs >= bins[i]) & (confs <= bins[i + 1])
        else:
            mask = (confs > bins[i]) & (confs <= bins[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / len(rows)) * abs(float(correct[mask].mean()) - float(confs[mask].mean()))
    return float(total)


def mean_defined(rows: list[dict], key: str, flag: str) -> float:
    picked = [r[key] for r in rows if r[flag]]
    if not picked:
        return 0.0
    return float(np.mean(picked))


def summarize(rows: list[dict]) -> dict:
    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    by_type = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row["f1"])
    for name, vals in grouped.items():
        by_type[name] = float(np.mean(vals))
    return {
        "n": len(rows),
        "em": avg("em"),
        "f1": avg("f1"),
        "probe_em": avg("probe_em"),
        "probe_f1": avg("probe_f1"),
        "hold": avg("hold"),
        "prop": mean_defined(rows, "prop", "prop_defined"),
        "admit": mean_defined(rows, "admit", "admit_defined"),
        "carry": mean_defined(rows, "carry", "carry_defined"),
        "ece": ece_score(rows),
        "checks": avg("checks"),
        "used_tokens": avg("used_tokens"),
        "transmitted_tokens": avg("transmitted_tokens"),
        "by_type": by_type,
    }


def mean_over_seeds(per_seed: dict) -> dict:
    out = {}
    settings = next(iter(per_seed.values())).keys()
    for setting in settings:
        out[setting] = {}
        methods = per_seed[next(iter(per_seed))][setting].keys()
        for method in methods:
            rows = [per_seed[seed][setting][method] for seed in per_seed]
            keys = ["em", "f1", "probe_em", "probe_f1", "hold", "prop", "admit", "carry", "ece", "checks", "used_tokens", "transmitted_tokens"]
            packed = {}
            for key in keys:
                vals = [r[key] for r in rows]
                packed[key] = float(np.mean(vals))
                packed[key + "_std"] = float(np.std(vals))
            packed["n"] = rows[0]["n"]
            packed["by_type"] = rows[0]["by_type"]
            out[setting][method] = packed
    return out


FULL_GATES = {
    "prov": True,
    "temp": True,
    "entail": True,
    "request": True,
    "memory": True,
    "qscore": True,
    "closure": True,
    "contradict": True,
    "retain": True,
    "attest": True,
}


def tune(models: Models, dev: list, base_cfg: dict) -> dict:
    grid = []
    for theta_app in (0.08, 0.14, 0.22):
        for theta_q in (0.10, 0.25):
            cfg = dict(base_cfg)
            cfg["theta_app"] = theta_app
            cfg["theta_q"] = theta_q
            print(f"tune app={theta_app:.2f} q={theta_q:.2f}", flush=True)
            result = evaluate_split(
                models,
                dev,
                cfg,
                seeds=[0],
                settings=["mixed"],
                methods=["trustmem", "marag"],
                gates=FULL_GATES,
            )
            stats = result["mean"]["mixed"]["trustmem"]
            score = 0.5 * (stats["f1"] + stats["probe_f1"]) - 0.2 * stats["prop"]
            grid.append((score, cfg, stats))
            print(
                f"  f1={stats['f1']:.3f} probe={stats['probe_f1']:.3f} prop={stats['prop']:.3f} score={score:.3f}",
                flush=True,
            )
    grid.sort(key=lambda item: item[0], reverse=True)
    best_cfg = grid[0][1]
    return {"best": {k: best_cfg[k] for k in ("theta_app", "theta_q", "theta_entail", "margin")}, "score": grid[0][0]}


def fmt_table(block: dict) -> str:
    lines = []
    for setting, methods in block.items():
        lines.append(f"[{setting}]")
        for method, stats in methods.items():
            lines.append(
                f"  {method:10s} EM {stats['em']:.3f} F1 {stats['f1']:.3f} "
                f"pF1 {stats['probe_f1']:.3f} Hold {stats['hold']:.3f} Prop {stats['prop']:.3f} Adm {stats['admit']:.3f} "
                f"ECE {stats['ece']:.3f} Carry {stats['carry']:.3f} "
                f"Chk {stats['checks']:.2f} Tok {stats['used_tokens']:.1f}"
            )
    return "\n".join(lines)


def run_review(models: Models, base_cfg: dict, n_test: int) -> None:
    global RESULT_PATH
    RESULT_PATH = ROOT / "results" / "attest_results.json"
    methods = ["marag", "crag", "trustmem"]
    output = {"config": base_cfg, "stage": "review", "protocol": {
        "cap": CAP,
        "delay": DELAY,
        "attest": "peer search of own pages plus cross-producer sentence filter",
        "crag": "one corrective retrieval when the quote is absent from the cited page",
        "seeds": "distractor entity only",
    }}
    trace = []
    output["datasets"] = {}
    for name in ("hotpot", "2wiki"):
        print("dataset", name, flush=True)
        HONEST_CACHE.clear()
        _, test = load_examples(name, 36, n_test, seed=3152)
        print("n_test", len(test), flush=True)
        result = evaluate_split(
            models,
            test,
            base_cfg,
            seeds=[0, 1, 2],
            settings=["clean", "mixed", *FAULTS],
            methods=methods,
            gates=FULL_GATES,
            trace_bucket=trace if name == "hotpot" else None,
        )
        output["datasets"][name] = result
        print(fmt_table(result["mean"]), flush=True)
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("ablation", flush=True)
    HONEST_CACHE.clear()
    _, test = load_examples("hotpot", 36, n_test, seed=3152)
    variants = {
        "full": FULL_GATES,
        "wo_attest": {**FULL_GATES, "attest": False},
        "wo_retain": {**FULL_GATES, "retain": False},
        "wo_request": {**FULL_GATES, "request": False},
    }
    output["ablation"] = {}
    for name, gates in variants.items():
        print("variant", name, flush=True)
        result = evaluate_split(
            models,
            test,
            base_cfg,
            seeds=[0, 1, 2],
            settings=["mixed", "conflict"],
            methods=["trustmem"],
            gates=gates,
        )
        output["ablation"][name] = {
            setting: result["mean"][setting]["trustmem"] for setting in ("mixed", "conflict")
        }
        RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
        for setting, stats in output["ablation"][name].items():
            print(
                f"  {setting} F1 {stats['f1']:.3f} pF1 {stats['probe_f1']:.3f} Hold {stats['hold']:.3f}",
                flush=True,
            )

    print("sweep", flush=True)
    HONEST_CACHE.clear()
    _, test = load_examples("hotpot", 36, n_test, seed=3152)
    output["sweep"] = []
    grid = [(4, 4), (8, 4), (16, 4), (32, 4), (8, 1), (8, 8)]
    for cap, delay in grid:
        print(f"sweep cap={cap} delay={delay}", flush=True)
        result = evaluate_split(
            models,
            test,
            base_cfg,
            seeds=[0, 1],
            settings=["mixed"],
            methods=["marag", "crag", "trustmem"],
            gates=FULL_GATES,
            cap=cap,
            delay=delay,
        )
        row = {"cap": cap, "delay": delay, "mixed": result["mean"]["mixed"]}
        output["sweep"].append(row)
        RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
        for method, stats in row["mixed"].items():
            print(
                f"  {method:10s} F1 {stats['f1']:.3f} pF1 {stats['probe_f1']:.3f} Hold {stats['hold']:.3f}",
                flush=True,
            )

    output["trace"] = trace
    RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (ROOT / "results" / "attest_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print("wrote", RESULT_PATH, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "tune", "eval", "ablate", "all", "final", "smoke", "innov", "review"], default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-dev", type=int, default=36)
    parser.add_argument("--n-test", type=int, default=180)
    args = parser.parse_args()
    if args.stage == "pilot":
        args.n_dev = 8
        args.n_test = 12
    device = args.device if torch.cuda.is_available() else "cpu"
    print("loading models on", device, flush=True)
    models = load_models(device)
    base_cfg = {
        "theta_entail": 0.20,
        "theta_app": 0.12,
        "theta_q": 0.15,
        "margin": 0.20,
        "entail": True,
    }
    trace = []
    output = {"config": base_cfg, "stage": args.stage}

    if args.stage == "review":
        n_test = 120 if args.n_test == 180 else args.n_test
        run_review(models, base_cfg, n_test)
        return

    if args.stage in {"tune", "all"}:
        dev, _ = load_examples("hotpot", args.n_dev, args.n_test, seed=3152)
        tuned = tune(models, dev, base_cfg)
        base_cfg.update(tuned["best"])
        output["tune"] = tuned
        output["config"] = base_cfg
        print("best", tuned, flush=True)

    global RESULT_PATH
    if args.stage in {"smoke", "innov"}:
        RESULT_PATH = ROOT / "results" / "closure_results.json"
    if args.stage == "pilot":
        args.n_dev = 8
        args.n_test = 12
    if args.stage == "smoke":
        args.n_dev = 8
        args.n_test = 16
    if args.stage in {"final", "innov"}:
        args.n_dev = 36
        args.n_test = 120
    if args.stage in {"pilot", "eval", "all", "final", "smoke", "innov"}:
        seeds = [0] if args.stage in {"pilot", "smoke"} else [0, 1, 2]
        if args.stage in {"pilot", "smoke"}:
            settings = ["clean", "mixed", "conflict"] if args.stage == "smoke" else ["clean", "mixed"]
        else:
            settings = ["clean", "mixed", *FAULTS]
        n_dev = 8 if args.stage in {"pilot", "smoke"} else 36
        n_test = args.n_test
        names = ("hotpot",) if args.stage == "smoke" else ("hotpot", "2wiki")
        output["datasets"] = {}
        for name in names:
            print("dataset", name, flush=True)
            HONEST_CACHE.clear()
            _, test = load_examples(name, n_dev, n_test, seed=3152)
            print("n_test", len(test), flush=True)
            result = evaluate_split(
                models,
                test,
                base_cfg,
                seeds=seeds,
                settings=settings,
                methods=METHODS,
                gates=FULL_GATES,
                trace_bucket=trace if name == "hotpot" else None,
            )
            output["datasets"][name] = result
            print(fmt_table(result["mean"]), flush=True)
            RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.stage in {"ablate", "all", "final", "innov"}:
        print("ablation", flush=True)
        HONEST_CACHE.clear()
        _, test = load_examples("hotpot", 36, args.n_test, seed=3152)
        variants = {
            "full": FULL_GATES,
            "wo_closure": {**FULL_GATES, "closure": False, "contradict": False},
            "wo_contradict": {**FULL_GATES, "contradict": False},
            "wo_retain": {**FULL_GATES, "retain": False},
            "wo_request": {**FULL_GATES, "request": False},
            "wo_memory": {**FULL_GATES, "memory": False},
        }
        output["ablation"] = {}
        for name, gates in variants.items():
            print("variant", name, flush=True)
            result = evaluate_split(
                models,
                test,
                base_cfg,
                seeds=[0] if args.stage == "smoke" else [0, 1, 2],
                settings=["mixed"],
                methods=["trustmem"],
                gates=gates,
            )
            output["ablation"][name] = result["mean"]["mixed"]["trustmem"]
            RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
            stats = output["ablation"][name]
            print(
                f"  F1 {stats['f1']:.3f} pF1 {stats['probe_f1']:.3f} Hold {stats['hold']:.3f}",
                flush=True,
            )

    output["trace"] = trace
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print("wrote", RESULT_PATH, flush=True)


if __name__ == "__main__":
    main()
