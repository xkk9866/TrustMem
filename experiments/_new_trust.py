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
    for message in messages:
        target = find_titled(docs, message["span"])
        if target is None or target["doc_id"] == message["doc_id"]:
            continue
        span, _score = answer_pages(reader, question, [target], corrupted)
        if span and not (asks and attached(span, question)):
            trace[0]["decision"] = "follow"
            trace[0]["answer"] = span
            return span, trace
    pool = named or [next(doc for doc in docs if doc["doc_id"] == messages[0]["doc_id"])] if messages else []
    if pool:
        span, _score = answer_pages(reader, question, pool[:2], corrupted)
        if span and not (asks and attached(span, question)):
            trace[0]["decision"] = "direct"
            trace[0]["answer"] = span
            return span, trace
    trace[0]["decision"] = "empty"
    return "", trace
