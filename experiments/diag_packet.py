"""Which four-sentence cut keeps the answer inside a stored packet."""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from run_trustmem import gold_recall, tokens

root = Path(__file__).resolve().parents[1]
with (root / "results" / "packets_pilot.pkl").open("rb") as handle:
    test_packets, _, meta = pickle.load(handle)


def rows(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or [{"text": contract.claim, "doc_id": "", "title": ""}]:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                out.append((contract.producer, ev.get("doc_id", ""), text))
    return out


def quota(question, items, k=4, per=2):
    """Keep the best sentences inside each producer so the second hop is not crowded out."""
    groups = {}
    for producer, doc_id, text in items:
        groups.setdefault(producer, []).append(text)
    chosen = []
    ranked_groups = []
    q = tokens(question)
    for producer, sents in groups.items():
        ordered = bm25_top(question, sents, k=len(sents))
        ranked_groups.append(ordered)
    # round-robin across producers
    for depth in range(per):
        for ordered in ranked_groups:
            if depth < len(ordered):
                chosen.append(ordered[depth])
            if len(chosen) >= k:
                return chosen[:k]
    return chosen[:k]


def sentences(stored):
    out = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or [{"text": contract.claim}]:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def bm25_top(question, sents, k=4):
    docs = [re.findall(r"[a-z0-9]+", text.lower()) or ["empty"] for text in sents]
    ranker = BM25Okapi(docs)
    query = re.findall(r"[a-z0-9]+", question.lower()) or ["unknown"]
    order = sorted(range(len(sents)), key=lambda i: float(ranker.get_scores(query)[i]), reverse=True)
    return [sents[i] for i in order[:k]]


def density_top(question, sents, k=4):
    q = tokens(question)
    def score(text):
        toks = tokens(text)
        hit = len(q & toks)
        return hit / (len(toks) ** 0.5) if toks else 0.0
    return sorted(sents, key=score, reverse=True)[:k]


def bridge_top(question, sents, k=4):
    """Prefer a sentence that names a token shared by two other sentences and absent from the question."""
    q = tokens(question)
    sets = [tokens(text) for text in sents]
    shared = set()
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            shared |= left & right
    shared -= q
    def score(text):
        toks = tokens(text)
        bridge = len(toks & shared)
        density = (len(q & toks) / (len(toks) ** 0.5)) if toks else 0.0
        return (bridge, density)
    return sorted(sents, key=score, reverse=True)[:k]


def hop_cut(question, items, k=4):
    """Hop 1 is the sentence closest to the question. Hop 2 names an entity from hop 1 that the question does not."""
    by_doc = {}
    for _, doc_id, text in items:
        by_doc.setdefault(doc_id or text[:48], []).append(text)
    doc_ranked = []
    for doc_id, sents in by_doc.items():
        ordered = bm25_top(question, sents, k=len(sents))
        doc_ranked.append(ordered)
    if not doc_ranked:
        return []
    # rank documents by how well their best sentence matches the question
    def lead_score(ordered):
        q = tokens(question)
        toks = tokens(ordered[0])
        return (len(q & toks) / (len(toks) ** 0.5)) if toks else 0.0
    doc_ranked.sort(key=lead_score, reverse=True)
    hop1 = doc_ranked[0][0]
    bridge = tokens(hop1) - tokens(question)
    chosen = [hop1]
    rest = []
    for ordered in doc_ranked[1:]:
        for text in ordered:
            if text in chosen:
                continue
            overlap = len(tokens(text) & bridge)
            rest.append((overlap, lead_score([text]), text))
    rest.sort(key=lambda row: (row[0], row[1]), reverse=True)
    for _, _, text in rest:
        if text not in chosen:
            chosen.append(text)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        for ordered in doc_ranked:
            for text in ordered:
                if text not in chosen:
                    chosen.append(text)
                if len(chosen) >= k:
                    return chosen[:k]
    return chosen[:k]


def link_cut(question, stored, k=4):
    """Keep a sentence that names another admitted page, plus the lead sentence of every page named by the question or by such a sentence."""
    rows = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            title = ev.get("title", "") or ""
            if text and text not in seen:
                seen.add(text)
                rows.append((title, text))
    if not rows:
        return []
    titles = sorted({title for title, _ in rows if title})
    q = tokens(question)

    def title_toks(title):
        return tokens(title)

    anchored = {title for title in titles if title_toks(title) & q}
    mentions = []
    named = set()
    for title, text in rows:
        text_toks = tokens(text)
        for other in titles:
            if other == title:
                continue
            other_toks = title_toks(other)
            if other_toks and other_toks <= text_toks:
                mentions.append(text)
                named.add(other)
                break
    keep_titles = anchored | named
    chosen = []
    for title, text in rows:
        if text in mentions and text not in chosen:
            chosen.append(text)
    # one lead sentence from each kept page, lead = highest question overlap
    by_title = {}
    for title, text in rows:
        by_title.setdefault(title, []).append(text)
    leads = []
    for title in keep_titles:
        sents = by_title.get(title, [])
        if not sents:
            continue
        sents = sorted(sents, key=lambda text: len(q & tokens(text)), reverse=True)
        leads.append(sents[0])
    for text in leads + [text for _, text in rows]:
        if text not in chosen:
            chosen.append(text)
        if len(chosen) >= k:
            break
    return chosen[:k]


def consistent_cut(question, stored, k=4):
    """Drop pages that share no entity with a second page. Rank only what remains."""
    rows = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                rows.append(ev)
    if not rows:
        return []
    q = tokens(question)
    by_doc = {}
    for ev in rows:
        by_doc.setdefault(ev.get("doc_id", ev.get("text", "")[:24]), []).append(ev)
    doc_toks = {}
    titles = {}
    for doc_id, evs in by_doc.items():
        blob = " ".join(f"{ev.get('title', '')} {ev.get('text', '')}" for ev in evs)
        doc_toks[doc_id] = tokens(blob) - q
        titles[doc_id] = tokens(evs[0].get("title", ""))
    counts = {}
    for toks in doc_toks.values():
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
    bridge = {tok for tok, n in counts.items() if n >= 2 and len(tok) > 3}
    # a title token named inside a different page is also a bridge
    for doc_id, evs in by_doc.items():
        for other, title_toks in titles.items():
            if other == doc_id or not title_toks:
                continue
            page = tokens(" ".join(ev.get("text", "") for ev in evs))
            if title_toks <= page:
                bridge |= title_toks - q
    kept = []
    for doc_id, evs in by_doc.items():
        page = tokens(" ".join(f"{ev.get('title', '')} {ev.get('text', '')}" for ev in evs))
        if bridge and (page & bridge or titles[doc_id] & bridge):
            kept.extend(ev.get("text", "") for ev in evs)
    if not kept:
        kept = [ev.get("text", "") for ev in rows]
    return bm25_top(question, kept, k=k)


def anchored_cut(question, stored, k=4):
    """Keep pages in the question-anchored proper-name component, then the best sentences on those pages."""
    rows = []
    seen = set()
    for contract in stored:
        for ev in contract.evidence or []:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                rows.append(ev)
    if not rows:
        return []
    by_doc = {}
    for ev in rows:
        key = ev.get("doc_id") or ev.get("title") or ev.get("text", "")[:40]
        by_doc.setdefault(key, []).append(ev)

    def page_body(evs):
        return " ".join(ev.get("text", "") for ev in evs).lower()

    doc_ids = list(by_doc)
    body = {doc_id: page_body(by_doc[doc_id]) for doc_id in doc_ids}
    q = tokens(question)
    neighbors = {doc_id: set() for doc_id in doc_ids}
    for left in doc_ids:
        for right in doc_ids:
            if left >= right:
                continue
            title_l = (by_doc[left][0].get("title") or "").strip()
            title_r = (by_doc[right][0].get("title") or "").strip()
            hit = False
            if len(title_l) > 3 and title_l.lower() in body[right]:
                hit = True
            if len(title_r) > 3 and title_r.lower() in body[left]:
                hit = True
            if hit:
                neighbors[left].add(right)
                neighbors[right].add(left)
    anchors = []
    for doc_id, evs in by_doc.items():
        if tokens(evs[0].get("title", "")) & q:
            anchors.append(doc_id)
    keep = set()
    for doc_id in anchors:
        if neighbors[doc_id]:
            keep.add(doc_id)
            keep |= neighbors[doc_id]
    if not keep:
        # no titled anchor; keep the largest linked component
        seen_nodes = set()
        best = []
        for doc_id in doc_ids:
            if doc_id in seen_nodes or not neighbors[doc_id]:
                continue
            stack = [doc_id]
            comp = []
            while stack:
                node = stack.pop()
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                comp.append(node)
                stack.extend(neighbors[node])
            if len(comp) > len(best):
                best = comp
        keep = set(best)
    if not keep:
        kept_text = [ev.get("text", "") for ev in rows]
    else:
        kept_text = [ev.get("text", "") for doc_id in keep for ev in by_doc[doc_id]]
    return kept_text


for setting in ("mixed", "unsupported", "incomplete", "conflict"):
    print("\n==", setting)
    for method in ("crag", "crag_plus", "trustmem"):
        acc = {name: 0.0 for name in ("all", "anchor", "anchor4")}
        n = an = count = 0
        for index, stored in enumerate(test_packets[setting][method]):
            sents = sentences(stored)
            count += 1
            if not sents:
                continue
            answer = meta[index]["answer"]
            question = meta[index]["question"]
            n += len(sents)
            kept = anchored_cut(question, stored)
            an += len(kept)
            acc["all"] += gold_recall(" ".join(sents), answer)
            acc["anchor"] += gold_recall(" ".join(kept), answer)
            acc["anchor4"] += gold_recall(" ".join(bm25_top(question, kept, k=4) if kept else []), answer)
        bits = " ".join(f"{name}={acc[name]/count:.3f}" for name in acc)
        print(f"  {method:10s} n={n/count:.1f} anchor_n={an/count:.1f} {bits}")
