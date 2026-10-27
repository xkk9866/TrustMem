"""Shared pieces: answer normalisation, page partition, rendering, prompts."""
from __future__ import annotations

import json
import math
import random
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ---------------------------------------------------------------- answers
def normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    if normalize(pred) in ("yes", "no", "noanswer") or normalize(gold) in ("yes", "no", "noanswer"):
        return float(normalize(pred) == normalize(gold))
    common = {}
    for tok in p:
        common[tok] = common.get(tok, 0) + 1
    same = 0
    for tok in g:
        if common.get(tok, 0):
            same += 1
            common[tok] -= 1
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def score_item(pred: str, item: dict) -> dict:
    golds = [item["answer"], *item.get("aliases", [])]
    return {"f1": max(f1(pred, g) for g in golds), "em": max(em(pred, g) for g in golds)}


def clean_answer(text: str) -> str:
    """First line of a model reply, without a leading label or trailing period."""
    text = re.sub(r"(?is)<think>.*?</think>", " ", str(text)).strip()
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"(?i)^(final answer|answer|intermediate answer|so the final answer is)\s*[:：]\s*", "", line)
    line = re.sub(r"^\[agent \d+\]\s*", "", line)
    line = line.strip().strip("\"'`*").strip()
    if line.endswith(".") and not re.search(r"\b[A-Z]\.$", line):
        line = line[:-1]
    return " ".join(line.split()[:20])


ABSTAIN = re.compile(
    r"\b(not (contain|mention|provide|specif|state|includ|given|availab)\w*|no (information|mention)|cannot|can ?not|unknown|unclear|insufficient)\b"
)


def is_none(text: str) -> bool:
    key = normalize(text)
    if key in ("", "none", "unknown", "not stated", "not mentioned", "no answer", "noanswer", "n/a", "na"):
        return True
    return key.startswith("none ") or bool(ABSTAIN.search(key))


# ---------------------------------------------------------------- data
def load(dataset: str, n: int, offset: int = 0) -> list[dict]:
    rows = []
    with open(DATA / f"{dataset}.jsonl", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if len(rows) >= n:
                break
            rows.append(json.loads(line))
    return rows


def partition(item: dict, k: int, seed: int, mirror: int = 1) -> list[list[dict]]:
    """Deal the question's pages to k specialists after a shuffle keyed by the question.

    With mirror=r, the pages are dealt to k/r groups and every group is held by r
    specialists (replicas); specialists r*g .. r*g+r-1 hold group g."""
    docs = list(item["docs"])
    key = f"{item['id']}:{k}:{seed}" if mirror == 1 else f"{item['id']}:{k}:{seed}:r{mirror}"
    random.Random(key).shuffle(docs)
    n = k // mirror
    groups = [[] for _ in range(n)]
    for index, doc in enumerate(docs):
        groups[index % n].append(doc)
    return [list(g) for g in groups for _ in range(mirror)]


def render_pages(docs: list[dict], limit: int = 1400) -> str:
    blocks = []
    for doc in docs:
        text = doc["text"]
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"## {doc['title']}\n{text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- prompts
SPECIALIST = """You are a specialist agent. You can see only the pages below.

{pages}

Question: {question}
Reply with only the answer: a short span such as a name, date, place or number, copied from the pages when possible. If these pages do not contain the answer, reply NONE."""

PROPOSE = """You are a specialist agent. You can see only the pages below.

{pages}

Question: {question}
Give your best answer from these pages, even if they only suggest it. Reply with only the answer, as a short span such as a name, date, place or title, copied from the pages when possible. Reply yes or no only if the question asks for yes or no."""

VERIFY = """You are a specialist agent. You can see only the pages below.

{pages}

Claim: the answer to the question "{question}" is "{value}".
Do these pages state this? Reply Yes or No."""

# Eight domain-neutral demonstrations, shared by every method that decomposes a question.
DEMOS = [
    {
        "q": "When was the founder of craigslist born?",
        "plan": [("LOOKUP", "Who founded craigslist?"), ("LOOKUP", "When was #1 born?")],
        "vals": ["Craig Newmark", "December 6, 1952"],
        "notes": ["Craigslist was founded in 1995 by Craig Newmark.", "Craig Newmark (born December 6, 1952) is an American internet entrepreneur."],
        "answer": "December 6, 1952",
    },
    {
        "q": "Who is older, Barack Obama or Angela Merkel?",
        "plan": [("LOOKUP", "When was Barack Obama born?"), ("LOOKUP", "When was Angela Merkel born?"), ("REASON", "Who is older, Barack Obama or Angela Merkel?")],
        "vals": ["August 4, 1961", "July 17, 1954", "Angela Merkel"],
        "notes": ["Barack Obama (born August 4, 1961) is an American politician.", "Angela Merkel (born July 17, 1954) is a German politician."],
        "answer": "Angela Merkel",
    },
    {
        "q": "Which film has the director who was born later, Jaws or The Godfather?",
        "plan": [("LOOKUP", "Who directed Jaws?"), ("LOOKUP", "When was #1 born?"), ("LOOKUP", "Who directed The Godfather?"), ("LOOKUP", "When was #3 born?"), ("REASON", "Which film has the director who was born later, Jaws or The Godfather?")],
        "vals": ["Steven Spielberg", "December 18, 1946", "Francis Ford Coppola", "April 7, 1939", "Jaws"],
        "notes": ["Jaws is a 1975 American thriller film directed by Steven Spielberg.", "Steven Spielberg (born December 18, 1946) is an American filmmaker.", "The Godfather is a 1972 American crime film directed by Francis Ford Coppola.", "Francis Ford Coppola (born April 7, 1939) is an American filmmaker."],
        "answer": "Jaws",
    },
    {
        "q": "Who was the maternal grandfather of George Washington?",
        "plan": [("LOOKUP", "Who was the mother of George Washington?"), ("LOOKUP", "Who was the father of #1?")],
        "vals": ["Mary Ball Washington", "Joseph Ball"],
        "notes": ["George Washington was the son of Augustine Washington and Mary Ball Washington.", "Mary Ball Washington was the daughter of Joseph Ball."],
        "answer": "Joseph Ball",
    },
    {
        "q": "Are the directors of Jaws and Casino Royale from the same country?",
        "plan": [("LOOKUP", "Who directed Jaws?"), ("LOOKUP", "Which country is #1 from?"), ("LOOKUP", "Who directed Casino Royale?"), ("LOOKUP", "Which country is #3 from?"), ("REASON", "Are the directors of Jaws and Casino Royale from the same country?")],
        "vals": ["Steven Spielberg", "United States", "Martin Campbell", "New Zealand", "no"],
        "notes": ["Jaws is a 1975 American thriller film directed by Steven Spielberg.", "Steven Spielberg is an American filmmaker.", "Casino Royale (2006) was directed by Martin Campbell.", "Martin Campbell is a New Zealand film director."],
        "answer": "no",
    },
    {
        "q": "In which city is the headquarters of the company that makes the iPhone?",
        "plan": [("LOOKUP", "Which company makes the iPhone?"), ("LOOKUP", "In which city is the headquarters of #1?")],
        "vals": ["Apple", "Cupertino"],
        "notes": ["The iPhone is a line of smartphones made by Apple.", "Apple Inc. is headquartered in Cupertino, California."],
        "answer": "Cupertino",
    },
    {
        "q": "What is the capital of the country where the composer of Finlandia was born?",
        "plan": [("LOOKUP", "Who composed Finlandia?"), ("LOOKUP", "In which country was #1 born?"), ("LOOKUP", "What is the capital of #2?")],
        "vals": ["Jean Sibelius", "Finland", "Helsinki"],
        "notes": ["Finlandia is a tone poem by Jean Sibelius.", "Jean Sibelius was born in Hameenlinna, Finland.", "Helsinki is the capital of Finland."],
        "answer": "Helsinki",
    },
    {
        "q": "In what year was the Eiffel Tower completed?",
        "plan": [("LOOKUP", "In what year was the Eiffel Tower completed?")],
        "vals": ["1889"],
        "notes": ["The Eiffel Tower was completed in 1889."],
        "answer": "1889",
    },
]


def _planner_demos() -> str:
    blocks = []
    for d in DEMOS:
        lines = [f"Question: {d['q']}"]
        for i, (kind, text) in enumerate(d["plan"], start=1):
            lines.append(f"#{i} = {kind}: {text}")
        lines.append(f"ANSWER: #{len(d['plan'])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _self_ask_demos() -> str:
    blocks = []
    for j, d in enumerate(DEMOS):
        lines = [f"Question: {d['q']}", "Are follow up questions needed here: Yes."]
        vals = {}
        for i, ((kind, text), val) in enumerate(zip(d["plan"], d["vals"]), start=1):
            if kind != "LOOKUP":
                continue
            q = re.sub(r"#(\d+)", lambda m: vals[int(m.group(1))], text)
            lines += [f"Follow up: {q}", f"Agent replies: [agent {(i + j) % 2 + 1}] {val}", f"Intermediate answer: {val}"]
            vals[i] = val
        lines.append(f"So the final answer is: {d['answer']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _ircot_demos() -> str:
    blocks = []
    for d in DEMOS:
        notes = "\n".join(f"- {n}" for n in d["notes"])
        reasoning = " ".join(d["notes"]) + f" So the answer is: {d['answer']}"
        blocks.append(f"Question: {d['q']}\nNotes:\n{notes}\nReasoning: {reasoning}")
    return "\n\n".join(blocks)


def _inquiry_demos() -> str:
    blocks = []
    for d in DEMOS:
        lines = [f"Question: {d['q']}"]
        vals = {}
        for i, ((kind, text), val) in enumerate(zip(d["plan"], d["vals"]), start=1):
            if kind != "LOOKUP":
                continue
            q = re.sub(r"#(\d+)", lambda m: vals[int(m.group(1))], text)
            lines += [f"Follow up: {q}", f"Intermediate answer: {val}"]
            vals[i] = val
        lines.append(f"So the final answer is: {d['answer']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# The composer of TrustMem asks one follow up at a time, but it only ever sees the facts of
# one hypothetical derivation, never the claims in the shared memory.
INQUIRY = (
    "Answer the question by asking follow up questions. Each follow up is a simple question about one entity "
    "that a single encyclopedia page can answer. Other agents look up each follow up. "
    "When the facts are enough to answer, write the final answer.\n\n"
    + _inquiry_demos()
    + "\n\nQuestion: {question}\n"
)

PLANNER = (
    "Break the question into steps. Each step is a LOOKUP (a simple question about one entity that a single encyclopedia page can answer) "
    "or a REASON (a step that only uses earlier results, such as comparing two dates or checking whether two values are the same). "
    "Write the result of step i as #i, and in later steps refer to it as #i instead of describing it again. "
    "Finish with ANSWER: #k, the step that answers the question. "
    "A question that can be looked up directly is a single LOOKUP.\n\n"
    + _planner_demos()
    + "\n\nQuestion: {question}\n"
)

REASON = """Facts found so far:
{facts}

Question: {question}
Answer the question using the facts. Reply with only the answer. If the question asks which of two or more named people or things, reply with one of those names. Reply yes or no only if the question asks for yes or no."""

CLOSED_BOOK = """Question: {question}
Answer from your own knowledge and give your best guess if you are unsure. Do not reply no or unknown unless the question asks for yes or no. Reply with only the answer, as a short span such as a name, date, place or title. Reply yes or no only if it is a yes/no question."""

SELF_ASK = (
    "Answer the question by asking follow up questions. Other agents answer each follow up from their own pages; "
    "you see their replies and write the intermediate answer.\n\n"
    + _self_ask_demos()
    + "\n\nQuestion: {question}\nAre follow up questions needed here:"
)

IRCOT = (
    "Answer the question by reasoning one sentence at a time. After each sentence, other agents add notes from their own pages. "
    'When you know the answer, write "So the answer is: <short answer>".\n\n'
    + _ircot_demos()
    + "\n\nQuestion: {question}\nNotes:\n{notes}\nReasoning:{reasoning}"
)

BOARD = """Several agents posted replies to a question in a shared memory. Each agent can see different pages.

{board}

Question: {question}
Reply with only the answer, as a short span such as a name, date, place or title. Reply yes or no only if it is a yes/no question."""

JOINT = """{pages}
{notes}
Question: {question}
Reply with only the answer, as a short span such as a name, date, place or title. Reply yes or no only if it is a yes/no question."""

JOINT_COT = """{pages}
{notes}
Question: {question}
Think step by step in a few sentences, then finish with a line "Answer: <short answer>"."""

DEBATE = """You are a specialist agent. You can see only the pages below.

{pages}

Question: {question}
Other agents replied:
{others}
Using your pages and the other replies, give your final answer. Reply with only the answer, as a short span such as a name, date, place or title. Reply yes or no only if it is a yes/no question. Do not write an agent label. If you still cannot tell, reply with the most likely of the replies."""

ADVERSARY = """Question: {question}
Write a short answer that sounds plausible but is factually wrong. Use the same kind of answer (a person, a place, a date, a number, a title, or yes or no). Reply with only the wrong answer."""

ADVERSARY_ADAPTIVE = """Pages:
{pages}

Question: {question}
Copy from the pages a short span of the same kind as the answer (a name, date, place, number or title) that is NOT the correct answer to the question but could easily be mistaken for it. Reply with only the span."""

ADVERSARY_TARGET = """Pages:
{pages}

Question: {question}
Correct answer: {answer}
Copy from the pages a short span of the same kind as the correct answer (a name, date, place, number or title) that is NOT the correct answer but could easily be mistaken for it. Reply with only the span."""

ADVERSARY_NOTE = """Write one sentence that falsely states that the answer to the question below is {value}. It should read like a sentence from an encyclopedia page.

Question: {question}"""


# ---------------------------------------------------------------- plans
STEP_RE = re.compile(r"^#(\d+)\s*=\s*(LOOKUP|REASON)\s*:\s*(.+)$", re.I)
ANS_RE = re.compile(r"^ANSWER\s*:\s*#(\d+)", re.I)


def parse_plan(text: str, question: str) -> list[dict]:
    steps = []
    final = None
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            if steps:
                break
            continue
        match = STEP_RE.match(line)
        if match:
            index = int(match.group(1))
            if index != len(steps) + 1:
                break
            steps.append({"kind": match.group(2).upper(), "text": match.group(3).strip()})
            continue
        match = ANS_RE.match(line)
        if match:
            final = int(match.group(1))
            break
        if line.lower().startswith("question"):
            break
    if not steps:
        return trivial_plan(question)
    if final is None or not 1 <= final <= len(steps):
        final = len(steps)
    steps = steps[:final]
    # a step may only refer to earlier steps
    for index, step in enumerate(steps, start=1):
        refs = [int(r) for r in re.findall(r"#(\d+)", step["text"])]
        if any(r >= index for r in refs):
            return trivial_plan(question)
        step["refs"] = refs
    if steps[0]["kind"] == "REASON":
        return trivial_plan(question)
    return steps


def trivial_plan(question: str) -> list[dict]:
    return [{"kind": "LOOKUP", "text": question, "refs": []}]


def plan_text(steps: list[dict]) -> str:
    lines = [f"#{i} = {s['kind']}: {s['text']}" for i, s in enumerate(steps, start=1)]
    lines.append(f"ANSWER: #{len(steps)}")
    return "\n".join(lines)


def instantiate(template: str, values: dict[int, str]) -> str:
    return re.sub(r"#(\d+)", lambda m: values.get(int(m.group(1)), m.group(0)), template)


def logsumexp(xs: list[float]) -> float:
    if not xs:
        return -math.inf
    top = max(xs)
    if top == -math.inf:
        return top
    return top + math.log(sum(math.exp(x - top) for x in xs))
