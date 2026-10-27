"""TrustMem: shared memory carries a substituted question, not a sentence.

Each specialist may copy one span from a page it holds. The composer never
receives page text. It may return one of the spans it was sent, or it may
name one message. In the latter case the quoted phrase in the question is
replaced by that span and the new question is read on the remaining pages.
The final answer has to be one of those spans or the title of a page that
sent one.

Baselines use the same model and the same six-sentence page budget:
joint reading, IRCoT, Self-Ask, open specialist answers, and carrying the
original question onto the page a span names.
"""
from __future__ import annotations

import json
import os
import random
import re
import string
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "results" / "subst_live.jsonl"
OUT = ROOT / "results" / "subst_read.json"
PROBE_OUT = ROOT / "results" / "subst_probe.json"
MODEL_NAME = os.environ.get("TRUSTMEM_LLM", "Qwen/Qwen2.5-3B-Instruct")
MAX_ROUNDS = 4
FAKE_ANSWER = "Marceline Duval"

SPECIALIST = """Page title: {title}
Page text: {text}
{note}
Question: {question}

The question names this page. Copy the shortest person, date, or place on the page that the question depends on, even when the page does not contain the final answer. Also copy the words of the question that this fact identifies.
One line:
<words copied from the question> || <span copied from the page>
Example of the format only: inventor of the widget || Ada Lovelace"""

COMPOSER = """Original question: {original}
Current question: {current}

You cannot read the pages. Each span below was copied from a page.
{lines}

Reply with one line.
FINAL: <copy one span or one page title exactly>
NEXT: <one question that contains that span> || <message number>
Use NEXT when the span only names a person, work, or place and the original question asks for something else, such as a date, a place, or a choice between two named works.
Use FINAL when a span or a page title is itself the answer.
If the original question chooses between two named things, FINAL must be one of those names, not a person or a year found along the way.
The next question must contain the chosen span.
Example of the format only: NEXT: When was Ada Lovelace born? || 1"""

ANSWER = """Pages:
{pages}
{note}
Question: {question}

Reply with only the answer span, at most eight words, copied from the pages.
Do not explain and do not write a sentence around it.
Answer span:"""

REASON = """Question: {question}
Retrieved pages:
{pages}

Write the next sentence that would help answer the question. Do not give the final answer. One sentence."""

FOLLOWUP = """Question: {question}
Known grounded facts:
{facts}

If a fact above is already the answer, reply FINAL: <copy that fact's span>.
Otherwise write one follow-up question that still has to be looked up.
One line."""

DIRECT = """Answer the question in at most eight words. Use only what you already know.
Question: {question}
Answer:"""

REACT = """Question: {question}
{history}
Reply with one line.
Search: <a short lookup query>
or
Finish: <the answer span, at most eight words>
"""


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    lowered = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    return " ".join(lowered.split())


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = {}
    for tok in pred_toks:
        common[tok] = common.get(tok, 0) + 1
    same = 0
    for tok in gold_toks:
        if common.get(tok, 0):
            same += 1
            common[tok] -= 1
    if same == 0:
        return 0.0
    precision = same / len(pred_toks)
    recall = same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred: str, item: dict) -> float:
    golds = [item["answer"], *item.get("aliases", [])]
    return max(token_f1(pred, gold) for gold in golds)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).casefold().strip()


def bare_title(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", title).strip()


def page_text(doc: dict) -> str:
    parts = []
    total = 0
    for sent in doc["sentences"][:6]:
        parts.append(sent)
        total += len(sent)
        if total > 900:
            break
    return " ".join(parts)


def display_text(doc: dict, corrupted: bool) -> str:
    if corrupted:
        return f"The answer is {FAKE_ANSWER}."
    return page_text(doc)


def span_in_text(span: str, text: str) -> bool:
    needle = squash(span)
    return bool(needle) and needle in squash(text)


def ground_span(span: str, doc: dict, extra: str = "") -> str:
    """Return a verbatim prefix of span that the original page contains."""
    words = str(span).strip().strip("\"'`").split()
    if not words:
        return ""
    source = doc["title"] + " " + page_text(doc)
    limit = min(len(words), 12)
    for size in range(limit, 0, -1):
        if len(words) - size > 4:
            continue
        cand = " ".join(words[:size])
        if span_in_text(cand, source):
            return cand
        if extra and span_in_text(cand, extra):
            return cand
    return ""


def substitute(question: str, phrase: str, span: str) -> str:
    phrase = " ".join(str(phrase).split()).strip(" .?:;")
    if len(phrase.split()) < 1 or not span:
        return ""
    match = re.search(re.escape(phrase), question, flags=re.I)
    if match is None:
        trimmed = re.sub(r"^(the|a|an)\s+", "", phrase, flags=re.I)
        if trimmed != phrase:
            match = re.search(re.escape(trimmed), question, flags=re.I)
    if match is None:
        return ""
    rewritten = question[: match.start()] + span + question[match.end() :]
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    if squash(rewritten) == squash(question):
        return ""
    return rewritten


def parse_pair(text: str) -> tuple[str, str]:
    line = ""
    for raw in str(text).splitlines():
        if raw.strip():
            line = raw.strip()
            break
    if not line or line.upper().startswith("NONE"):
        return "", ""
    if "||" not in line:
        return "", line.strip(" .")
    left, right = line.split("||", 1)
    return left.strip(" \"'`"), right.strip(" \"'`")


def parse_answer(text: str) -> str:
    cleaned = re.sub(r"(?is)</?span>|<think>.*?</think>", " ", str(text))
    line = ""
    for raw in cleaned.splitlines():
        if raw.strip():
            line = raw.strip()
            break
    line = re.sub(r"(?i)^(answer(?:\s+span)?|final)\s*:\s*", "", line).strip(" \"'`")
    if not line or line.upper().startswith("NONE"):
        return ""
    line = re.split(r"\b(?:Therefore|So the answer|Thus)\b", line)[0].strip(" ,;.")
    if len(line.split()) > 12:
        head = re.split(r"[.]", line)[0].strip()
        line = head or line
    return " ".join(line.split()[:12])


def parse_composer(text: str) -> tuple[str, str]:
    line = ""
    for raw in str(text).splitlines():
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
        if cleaned:
            line = cleaned
            break
    nxt = re.match(r"(?i)next\s*:\s*(.*?)\s*\|\|\s*(\d+)\s*$", line)
    if nxt:
        return "next", nxt.group(1).strip() + "||" + nxt.group(2)
    use = re.match(r"(?i)use\s*:\s*(\d+)", line)
    if use:
        return "use", use.group(1)
    final = re.sub(r"(?i)^final\s*:\s*", "", line).strip()
    return "final", final.strip(" \"'`")


def attached(fragment: str, question: str) -> bool:
    fragment = " ".join(str(fragment).split()).strip(" .?:;")
    if len(fragment) < 4:
        return False
    return re.search(re.escape(fragment), question, flags=re.I) is not None


def order_choice(question: str, messages: list, docs: list) -> str:
    """The question's own earlier/later word orders years copied from the named pages."""
    earlier = bool(re.search(r"\b(earlier|older|first|before)\b", question, flags=re.I))
    later = bool(re.search(r"\b(later|younger|after)\b", question, flags=re.I))
    if earlier == later:
        return ""
    death = bool(re.search(r"\b(died|dies|death)\b", question, flags=re.I)) and not bool(
        re.search(r"\bborn\b", question, flags=re.I)
    )
    scored = {}
    by_title = {doc["title"]: doc for doc in docs}
    for message in messages:
        title = bare_title(message["title"])
        if not attached(title, question):
            continue
        page = find_titled(docs, message["span"])
        if page is None or page["title"] == message["title"]:
            page = by_title.get(message["title"])
        if page is None:
            continue
        years = re.findall(r"\b(?:1[6-9][0-9]{2}|20[0-9]{2})\b", page_text(page))
        if not years:
            continue
        scored[title] = int(years[-1] if death else years[0])
    if len(scored) < 2:
        return ""
    return min(scored, key=scored.get) if earlier else max(scored, key=scored.get)


def admissible(message: dict, question: str) -> bool:
    """A page may speak only when the question names that page or the span."""
    if attached(bare_title(message["title"]), question):
        return True
    return attached(message["span"], question)


def find_titled(docs: list, span: str) -> dict | None:
    key = squash(bare_title(span))
    if len(key) < 3:
        return None
    for doc in docs:
        if squash(bare_title(doc["title"])) == key:
            return doc
    if len(key) < 8:
        return None
    hits = []
    for doc in docs:
        title = squash(bare_title(doc["title"]))
        if len(title) < 8:
            continue
        if title in key or key in title:
            hits.append(doc)
    if len(hits) == 1:
        return hits[0]
    return None


def match_allowed(text: str, messages: list) -> str:
    wanted = squash(text)
    if not wanted:
        return ""
    options = []
    for message in messages:
        options.append(message["span"])
        options.append(bare_title(message["title"]))
    for option in options:
        if squash(option) == wanted:
            return option
    contained = [option for option in options if squash(option) and squash(option) in wanted]
    if not contained:
        return ""
    return max(contained, key=lambda option: len(squash(option)))


def partition(docs: list, qid: str) -> list[list]:
    rng = random.Random(f"{qid}:2:random")
    ordered = list(docs)
    rng.shuffle(ordered)
    groups = [[], []]
    for index, doc in enumerate(ordered):
        groups[index % 2].append(doc)
    return groups


def corrupt_id(item: dict) -> str:
    for doc in item["docs"]:
        if doc["support"]:
            return doc["doc_id"]
    return item["docs"][0]["doc_id"] if item["docs"] else ""


class Reader:
    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.cache = {}
        self.tok.padding_side = "left"

    def _prompt(self, user: str) -> str:
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": user}], enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tok.apply_chat_template([{"role": "user", "content": user}], **kwargs)

    def _decode(self, new_ids, scores, row: int) -> tuple[str, float]:
        torch = self.torch
        tokens = []
        logps = []
        for index, token in enumerate(new_ids.tolist()):
            if token == self.tok.pad_token_id:
                continue
            if token == self.tok.eos_token_id:
                break
            tokens.append(token)
            if index < len(scores):
                dist = torch.log_softmax(scores[index][row].float(), dim=-1)
                logps.append(float(dist[token]))
        text = self.tok.decode(tokens, skip_special_tokens=True)
        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        mean_lp = sum(logps) / len(logps) if logps else -20.0
        return text, mean_lp

    def generate_many(self, users: list[str], max_new: int = 48) -> list[tuple[str, float]]:
        results = [None] * len(users)
        pending = []
        for index, user in enumerate(users):
            cached = self.cache.get((user, max_new))
            if cached is not None:
                results[index] = cached
            else:
                pending.append(index)
        if not pending:
            return results
        if len(pending) == 1:
            results[pending[0]] = self.generate(users[pending[0]], max_new)
            return results
        chunk = 4
        if len(pending) > chunk:
            for start in range(0, len(pending), chunk):
                part = pending[start : start + chunk]
                partial = self.generate_many([users[index] for index in part], max_new)
                for index, value in zip(part, partial):
                    results[index] = value
            return results
        torch = self.torch
        prompts = [self._prompt(users[index]) for index in pending]
        encoded = self.tok(prompts, return_tensors="pt", padding=True)
        input_ids = encoded.input_ids.to(self.device)
        attention = encoded.attention_mask.to(self.device)
        if input_ids.shape[1] > 5500:
            input_ids = input_ids[:, -5500:]
            attention = attention[:, -5500:]
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        new_tokens = out.sequences[:, input_ids.shape[1] :]
        for row, index in enumerate(pending):
            text, mean_lp = self._decode(new_tokens[row], out.scores, row)
            self.cache[(users[index], max_new)] = (text, mean_lp)
            results[index] = (text, mean_lp)
        return results

    def generate(self, user: str, max_new: int = 48) -> tuple[str, float]:
        key = (user, max_new)
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        encoded = self.tok(self._prompt(user), return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        if input_ids.shape[1] > 5500:
            input_ids = input_ids[:, -5500:]
        attention = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        text, mean_lp = self._decode(out.sequences[0, input_ids.shape[1] :], out.scores, 0)
        self.cache[key] = (text, mean_lp)
        return text, mean_lp


def render_pages(docs: list, corrupted: str = "") -> str:
    blocks = []
    for doc in docs:
        text = display_text(doc, corrupted and doc["doc_id"] == corrupted)
        blocks.append(f"{doc['title']}\n{text}")
    return "\n\n".join(blocks)


def answer_pages(reader: Reader, question: str, docs: list, corrupted: str = "", note: str = "") -> tuple[str, float]:
    if not docs:
        return "", -20.0
    shown = note if note else ""
    raw, score = reader.generate(
        ANSWER.format(pages=render_pages(docs, corrupted), question=question, note=shown)
    )
    parsed = parse_answer(raw)
    extra = FAKE_ANSWER if corrupted else ""
    best = ""
    for doc in docs:
        grounded = ground_span(parsed, doc, extra=extra if doc["doc_id"] == corrupted else "")
        if grounded:
            best = grounded
            break
    if corrupted and span_in_text(parsed, FAKE_ANSWER):
        best = FAKE_ANSWER
    return (best or parsed, score)


def bm25_pick(query: str, docs: list, k: int, blocked: set[str]) -> list:
    from rank_bm25 import BM25Okapi

    pool = [doc for doc in docs if doc["doc_id"] not in blocked]
    if not pool:
        return []
    corpus = []
    for doc in pool:
        corpus.append(re.findall(r"[a-z0-9]+", (doc["title"] + " " + page_text(doc)).lower()))
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not tokens or not any(corpus):
        return pool[:k]
    scores = BM25Okapi(corpus).get_scores(tokens)
    order = sorted(range(len(pool)), key=lambda index: float(scores[index]), reverse=True)
    return [pool[index] for index in order[:k]]


def run_joint(reader: Reader, item: dict) -> str:
    print("  joint pages", len(item["docs"]), flush=True)
    span, _ = answer_pages(reader, item["question"], item["docs"])
    print("  joint done", flush=True)
    return span


def run_open(reader: Reader, item: dict) -> tuple[str, dict | None]:
    groups = partition(item["docs"], item["id"])
    best = ("", -20.0, None)
    for group in groups:
        if not group:
            continue
        span, score = answer_pages(reader, item["question"], group)
        if span and score > best[1]:
            best = (span, score, None)
    return best[0], best[2]


def run_carry(reader: Reader, item: dict) -> str:
    groups = partition(item["docs"], item["id"])
    best_span, best_score, source = "", -20.0, None
    for group in groups:
        if not group:
            continue
        span, score = answer_pages(reader, item["question"], group)
        if span and score > best_score:
            owned = find_titled(group, span)
            best_span, best_score, source = span, score, owned
    target = find_titled(item["docs"], best_span)
    if target is None or (source is not None and target["doc_id"] == source["doc_id"]):
        return best_span
    followed, _ = answer_pages(reader, item["question"], [target])
    return followed or best_span


def run_direct(reader: Reader, item: dict) -> str:
    raw, _ = reader.generate(DIRECT.format(question=item["question"]), max_new=24)
    return parse_answer(raw)


def run_bm25(reader: Reader, item: dict) -> str:
    chosen = bm25_pick(item["question"], item["docs"], 3, set())
    span, _ = answer_pages(reader, item["question"], chosen)
    return span


def run_react(reader: Reader, item: dict) -> str:
    docs = item["docs"]
    history = []
    seen = set()
    chosen = []
    for _ in range(2):
        shown = "\n".join(history) if history else "No observations yet."
        raw, _ = reader.generate(
            REACT.format(question=item["question"], history=shown),
            max_new=40,
        )
        line = raw.strip().splitlines()[0] if raw.strip() else ""
        if line.lower().startswith("finish"):
            return parse_answer(re.sub(r"(?i)^finish\s*:\s*", "", line))
        query = re.sub(r"(?i)^search\s*:\s*", "", line).strip() or item["question"]
        picked = bm25_pick(query, docs, 1, seen)
        if not picked:
            break
        doc = picked[0]
        seen.add(doc["doc_id"])
        chosen.append(doc)
        history.append(f"Search: {query}")
        history.append(f"Observation: {doc['title']}: {page_text(doc)[:400]}")
    span, _ = answer_pages(reader, item["question"], chosen or docs[:1])
    return span


def run_ircot(reader: Reader, item: dict) -> str:
    docs = item["docs"]
    chosen = bm25_pick(item["question"], docs, 2, set())
    seen = {doc["doc_id"] for doc in chosen}
    for _ in range(2):
        raw, _ = reader.generate(
            REASON.format(question=item["question"], pages=render_pages(chosen)),
            max_new=40,
        )
        sentence = raw.strip().splitlines()[0] if raw.strip() else item["question"]
        extra = bm25_pick(sentence, docs, 1, seen)
        for doc in extra:
            if doc["doc_id"] not in seen:
                chosen.append(doc)
                seen.add(doc["doc_id"])
    span, _ = answer_pages(reader, item["question"], chosen)
    return span


def run_selfask(reader: Reader, item: dict) -> str:
    docs = item["docs"]
    facts = []
    chosen = []
    blocked = set()
    last_span = ""
    for _ in range(2):
        shown = "\n".join(facts) if facts else "None yet."
        raw, _ = reader.generate(
            FOLLOWUP.format(question=item["question"], facts=shown),
            max_new=40,
        )
        line = raw.strip().splitlines()[0] if raw.strip() else ""
        if line.upper().startswith("FINAL"):
            text = parse_answer(line)
            if text:
                return text
            line = ""
        query = line if len(line.split()) >= 3 else item["question"]
        picked = bm25_pick(query, docs, 1, blocked)
        if not picked:
            break
        blocked.add(picked[0]["doc_id"])
        chosen.append(picked[0])
        span, _ = answer_pages(reader, query, picked)
        if span:
            last_span = span
            facts.append(f"{query} {span}")
    if not chosen:
        return last_span
    span, _ = answer_pages(reader, item["question"], chosen)
    return span or last_span


def collect_messages(reader: Reader, question: str, docs: list, corrupted: str, trusting: bool, keep_all: bool = False) -> list:
    if not keep_all:
        named = [doc for doc in docs if attached(bare_title(doc["title"]), question)]
        if named:
            docs = named
    if not docs:
        return []
    prompts = []
    for doc in docs:
        text = display_text(doc, corrupted and doc["doc_id"] == corrupted)
        prompts.append(SPECIALIST.format(title=doc["title"], text=text, question=question, note=""))
    outputs = reader.generate_many(prompts)
    messages = []
    for doc, (raw, score) in zip(docs, outputs):
        phrase, span = parse_pair(raw)
        extra = FAKE_ANSWER if trusting and corrupted and doc["doc_id"] == corrupted else ""
        grounded = ground_span(span, doc, extra=extra)
        if not grounded:
            continue
        if not trusting and squash(grounded) == squash(FAKE_ANSWER):
            continue
        message = {
            "phrase": phrase,
            "span": grounded,
            "title": doc["title"],
            "doc_id": doc["doc_id"],
            "lp": score,
            "specialist": doc.get("owner", 0),
        }
        if keep_all or admissible(message, question):
            messages.append(message)
    return dedupe(messages)


def dedupe(messages: list) -> list:
    kept = {}
    for message in messages:
        key = squash(message["span"])
        previous = kept.get(key)
        mention = squash(bare_title(message["title"])) != key
        if previous is None:
            kept[key] = message
            continue
        previous_mention = squash(bare_title(previous["title"])) != key
        if mention and not previous_mention:
            kept[key] = message
        elif mention == previous_mention and message["lp"] > previous["lp"]:
            kept[key] = message
    ordered = sorted(kept.values(), key=lambda message: message["lp"], reverse=True)
    return ordered[:12]



GROUP = """Pages:
{pages}

Question: {question}

Copy the shortest person, date, or place that the question depends on.
One line:
<words from the question> || <span copied from the pages>
"""


def collect_groups(reader: Reader, question: str, docs: list, corrupted: str, trusting: bool) -> list:
    owners = {}
    for doc in docs:
        owners.setdefault(doc.get("owner", 0), []).append(doc)
    groups = [group for group in owners.values() if group]
    if not groups:
        return []
    prompts = [
        GROUP.format(pages=render_pages(group, corrupted if corrupted else ""), question=question)
        for group in groups
    ]
    outputs = reader.generate_many(prompts)
    messages = []
    for group, (raw, score) in zip(groups, outputs):
        phrase, span = parse_pair(raw)
        for doc in group:
            extra = FAKE_ANSWER if trusting and corrupted and doc["doc_id"] == corrupted else ""
            grounded = ground_span(span, doc, extra=extra)
            if not grounded:
                continue
            messages.append(
                {
                    "phrase": phrase,
                    "span": grounded,
                    "title": doc["title"],
                    "doc_id": doc["doc_id"],
                    "lp": score,
                    "specialist": doc.get("owner", 0),
                }
            )
            break
    return dedupe(messages)


def _asks_attribute(question: str) -> bool:
    return bool(re.match(r"(?i)(where|when|what|how|in what|in which)\b", question))


def run_trustmem(reader: Reader, item: dict, corrupted: str = "", trusting: bool = False) -> tuple[str, list]:
    """A span is an address. The addressed page answers the question.

    When the question itself asks which of two named things is earlier or later,
    the years on the pages those spans name are ordered by that word.
    """
    docs = []
    groups = partition(item["docs"], item["id"])
    for index, group in enumerate(groups):
        for doc in group:
            owned = dict(doc)
            owned["owner"] = index
            docs.append(owned)
    question = item["question"]
    named = [doc for doc in docs if attached(bare_title(doc["title"]), question)]
    if named:
        messages = collect_messages(reader, question, named, corrupted, trusting, keep_all=True)
    else:
        messages = []
    if not messages:
        messages = collect_groups(reader, question, docs, corrupted, trusting)
    ordered = order_choice(question, messages, docs)
    trace = [{"messages": [{"title": m["title"], "span": m["span"]} for m in messages]}]
    if ordered:
        trace[0]["decision"] = "order"
        return ordered, trace
    asks = _asks_attribute(question)
    seen = {message["doc_id"] for message in messages}
    frontier = list(messages)
    answer = ""
    for _hop in range(2):
        nxt = []
        for message in frontier:
            target = find_titled(docs, message["span"])
            if target is None or target["doc_id"] in seen:
                continue
            seen.add(target["doc_id"])
            span, _score = answer_pages(reader, question, [target], corrupted)
            if not span or (asks and attached(span, question)):
                continue
            answer = span
            nxt.append({"span": span, "doc_id": target["doc_id"], "title": target["title"], "phrase": ""})
        if not nxt:
            break
        frontier = nxt
        if all(find_titled(docs, message["span"]) is None for message in frontier):
            break
    if answer:
        trace[0]["decision"] = "follow"
        trace[0]["answer"] = answer
        return answer, trace
    pool = named or [next(doc for doc in docs if doc["doc_id"] == messages[0]["doc_id"])] if messages else []
    if pool:
        span, _score = answer_pages(reader, question, pool[:2], corrupted)
        if span and not (asks and attached(span, question)):
            trace[0]["decision"] = "direct"
            trace[0]["answer"] = span
            return span, trace
    trace[0]["decision"] = "empty"
    return "", trace

def self_check() -> None:
    question = "Where was the inventor of the widget born?"
    rewritten = substitute(question, "inventor of the widget", "Ada Lovelace")
    assert rewritten == "Where was the Ada Lovelace born?", rewritten
    assert substitute(question, "missing phrase", "Ada") == ""
    assert parse_pair("NONE") == ("", "")
    assert parse_pair("inventor of the widget || Ada Lovelace")[1] == "Ada Lovelace"
    kind, payload = parse_composer("NEXT: When was Ada Lovelace born? || 2")
    assert kind == "next" and payload.endswith("||2")
    doc = {"title": "Widget", "sentences": ["The widget was created by Ada Lovelace in London."]}
    assert ground_span("Ada Lovelace in London yesterday", doc) == "Ada Lovelace in London"
    assert match_allowed(
        "Ada Lovelace",
        [{"span": "Ada Lovelace", "title": "Widget"}],
    ) == "Ada Lovelace"
    print("self-check ok")


def parse_hotpot_wiki(example: dict) -> dict:
    titles = list(example["context"]["title"])
    groups = list(example["context"]["sentences"])
    support = set(example["supporting_facts"]["title"])
    docs = []
    for index, (title, sentences) in enumerate(zip(titles, groups)):
        clean = [re.sub(r"\s+", " ", str(sent)).strip() for sent in sentences]
        clean = [sent for sent in clean if sent]
        docs.append(
            {
                "doc_id": f"{index}:{title}",
                "title": title,
                "sentences": clean,
                "support": title in support,
            }
        )
    return {
        "id": str(example["id"]),
        "question": example["question"],
        "answer": example["answer"],
        "aliases": [],
        "type": example.get("type") or "unknown",
        "docs": docs,
    }


def parse_musique(example: dict) -> dict:
    support = {step["paragraph_support_idx"] for step in example["question_decomposition"]}
    docs = []
    for paragraph in example["paragraphs"]:
        pieces = re.split(r"(?<=[.!?])\s+", paragraph["paragraph_text"])
        clean = [re.sub(r"\s+", " ", piece).strip() for piece in pieces if piece.strip()]
        docs.append(
            {
                "doc_id": f"{paragraph['idx']}:{paragraph['title']}",
                "title": paragraph["title"],
                "sentences": clean or [paragraph["paragraph_text"][:900]],
                "support": paragraph["idx"] in support,
            }
        )
    return {
        "id": str(example["id"]),
        "question": example["question"],
        "answer": example["answer"],
        "aliases": list(example.get("answer_aliases") or []),
        "type": "musique",
        "docs": docs,
    }


def supported(item: dict) -> bool:
    blob = normalize_answer(" ".join(page_text(doc) for doc in item["docs"]))
    # Full pages, not the six-sentence budget: the filter matches the prior split.
    blob = normalize_answer(
        " ".join(sent for doc in item["docs"] for sent in doc["sentences"])
    )
    answer = normalize_answer(item["answer"])
    return bool(answer) and answer in blob and len(answer.split()) <= 12


def load_items(dataset: str, seed: int) -> tuple[list, list]:
    from datasets import load_dataset

    if dataset == "hotpot":
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
        parsed = []
        for example in raw:
            item = parse_hotpot_wiki(example)
            if item["type"] != "bridge" or not supported(item):
                continue
            parsed.append(item)
    elif dataset == "2wiki":
        raw = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        parsed = [parse_hotpot_wiki(example) for example in raw]
        parsed = [item for item in parsed if supported(item)]
    elif dataset == "musique":
        raw = load_dataset("dgslibisey/MuSiQue", split="validation")
        parsed = [parse_musique(example) for example in raw]
        parsed = [item for item in parsed if supported(item)]
    else:
        raise ValueError(dataset)
    rng = random.Random(seed)
    rng.shuffle(parsed)
    return parsed[:36], parsed[36:156]


def bootstrap(left: list, right: list) -> dict:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(3152)
    draws = rng.integers(0, len(diff), size=(2000, len(diff)))
    means = diff[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi)}


def done_keys() -> set[str]:
    found = set()
    if not LIVE.exists():
        return found
    for line in LIVE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found.add(f"{row['dataset']}|{row['seed']}|{row['id']}|{row['split']}")
    return found


def append_row(row: dict) -> None:
    LIVE.parent.mkdir(exist_ok=True)
    with LIVE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_item(reader: Reader, item: dict, with_note: bool) -> dict:
    preds = {
        "direct": run_direct(reader, item),
        "bm25": run_bm25(reader, item),
        "joint": run_joint(reader, item),
        "ircot": run_ircot(reader, item),
        "selfask": run_selfask(reader, item),
        "react": run_react(reader, item),
        "open": run_open(reader, item)[0],
        "carry": run_carry(reader, item),
    }
    trust_pred, trace = run_trustmem(reader, item)
    preds["trustmem"] = trust_pred
    if with_note:
        corrupted = corrupt_id(item)
        grounded, _ = run_trustmem(reader, item, corrupted=corrupted, trusting=False)
        trusting, _ = run_trustmem(reader, item, corrupted=corrupted, trusting=True)
        preds["noted"] = grounded
        preds["trusting"] = trusting
    scores = {name: best_f1(pred, item) for name, pred in preds.items()}
    return {"preds": preds, "scores": scores, "trace": trace}


def summarize(rows: list) -> dict:
    methods = sorted({key for row in rows for key in row["scores"]})
    columns = {name: [row["scores"][name] for row in rows] for name in methods}
    report = {
        "n": len(rows),
        "mean": {name: float(np.mean(columns[name])) for name in methods},
        "delta": {},
    }
    focus = "trustmem" if "trustmem" in columns else methods[0]
    for name in methods:
        if name == focus:
            continue
        report["delta"][f"{focus}-{name}"] = bootstrap(columns[focus], columns[name])
    if "noted" in columns and "trusting" in columns:
        report["delta"]["noted-trusting"] = bootstrap(columns["noted"], columns["trusting"])
    return report


def pick_device() -> str:
    import torch

    forced = os.environ.get("TRUSTMEM_DEVICE", "")
    if forced:
        return forced
    best = None
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        if best is None or free > best[0]:
            best = (free, index)
    if best is None or best[0] < 8 * 2**30:
        raise SystemExit("No GPU has 8 GiB free. Refusing to displace other jobs.")
    return f"cuda:{best[1]}"


def main() -> None:
    import pickle
    import traceback

    self_check()
    probe = os.environ.get("TRUSTMEM_PROBE") == "1"
    limit = int(os.environ.get("TRUSTMEM_LIMIT", "8" if probe else "120"))
    datasets = os.environ.get("TRUSTMEM_DATASETS", "2wiki,hotpot,musique").split(",")
    seeds = [3152] if probe else [3152, 2027]
    with_note = (not probe) and os.environ.get("TRUSTMEM_SKIP_NOTE") != "1"
    device = pick_device()
    print("device", device, "model", MODEL_NAME, "probe", probe, flush=True)
    cache = ROOT / "results" / "subst_slice.pkl"
    if os.environ.get("TRUSTMEM_PREP") == "1":
        planned = []
        for dataset in datasets:
            for seed in seeds:
                dev, test = load_items(dataset, seed)
                split = "dev" if probe else "test"
                items = (dev if probe else test)[:limit]
                planned.append((dataset, seed, split, items))
                print(f"prepared {dataset} seed {seed} {split} n={len(items)}", flush=True)
        cache.write_bytes(pickle.dumps(planned))
        print("cached", cache, flush=True)
        return
    if os.environ.get("TRUSTMEM_USE_CACHE") == "1" and cache.exists():
        planned = pickle.loads(cache.read_bytes())
        print("loaded cache", len(planned), flush=True)
    else:
        planned = []
        for dataset in datasets:
            for seed in seeds:
                dev, test = load_items(dataset, seed)
                split = "dev" if probe else "test"
                items = (dev if probe else test)[:limit]
                planned.append((dataset, seed, split, items))
                print(f"prepared {dataset} seed {seed} {split} n={len(items)}", flush=True)
        if os.environ.get("TRUSTMEM_PREP") == "1":
            cache.write_bytes(pickle.dumps(planned))
            print("cached", cache, flush=True)
            return
    import gc

    gc.collect()
    reader = Reader(device)
    finished = done_keys()
    grouped = {}
    t0 = time.time()
    for dataset, seed, split, items in planned:
            print(f"\n== {dataset} seed {seed} {split} n={len(items)}", flush=True)
            rows = []
            # Recover rows already written for this slice.
            if LIVE.exists():
                for line in LIVE.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["dataset"] == dataset and row["seed"] == seed and row["split"] == split:
                        rows.append(row)
            have = {row["id"] for row in rows}
            for item in items:
                if item["id"] in have:
                    continue
                started = time.time()
                print("  start", item["id"], flush=True)
                try:
                    result = evaluate_item(reader, item, with_note=with_note)
                except RuntimeError:
                    traceback.print_exc()
                    reader.torch.cuda.empty_cache()
                    continue
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "split": split,
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "preds": result["preds"],
                    "scores": result["scores"],
                    "trace": result["trace"],
                }
                append_row(row)
                rows.append(row)
                means = {name: float(np.mean([item_row["scores"][name] for item_row in rows])) for name in result["scores"]}
                print(
                    f"  {len(rows)}/{len(items)} {time.time() - started:.1f}s "
                    + " ".join(
                        f"{name} {means[name]:.3f}"
                        for name in ("trustmem", "joint", "ircot", "bm25", "selfask", "react", "direct")
                    )
                    + f" | pred {result['preds']['trustmem']!r} gold {item['answer']!r}",
                    flush=True,
                )
                if probe and len(rows) <= 2:
                    print("  TRACE", json.dumps(result["trace"], ensure_ascii=False)[:1500], flush=True)
            summary = summarize(rows)
            grouped.setdefault(dataset, {})[f"{split}-{seed}"] = summary
            print("  MEAN", {key: round(value, 3) for key, value in summary["mean"].items()}, flush=True)
            for key, delta in summary["delta"].items():
                mark = " *" if delta["lo"] > 0 else ""
                print(
                    f"  {key} {delta['mean']:+.3f} [{delta['lo']:+.3f}, {delta['hi']:+.3f}]{mark}",
                    flush=True,
                )
    payload = {
        "model": MODEL_NAME,
        "rounds": MAX_ROUNDS,
        "page_sentences": 6,
        "datasets": grouped,
        "elapsed_s": round(time.time() - t0, 1),
    }
    target = PROBE_OUT if probe else OUT
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", target, flush=True)


if __name__ == "__main__":
    main()
