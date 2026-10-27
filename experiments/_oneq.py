import os
import pickle
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "experiments")
import subst_read as s

planned = pickle.loads((s.ROOT / "results" / "subst_slice.pkl").read_bytes())
item = planned[0][3][0]
reader = s.Reader("cuda:0")
for doc in item["docs"]:
    if not s.attached(s.bare_title(doc["title"]), item["question"]):
        continue
    raw, _score = reader.generate(
        s.SPECIALIST.format(
            title=doc["title"],
            text=s.page_text(doc),
            question=item["question"],
            note="",
        )
    )
    phrase, span = s.parse_pair(raw)
    print("TITLE", doc["title"])
    print("RAW", repr(raw)[:400])
    print("GROUND", repr(s.ground_span(span, doc)))
