"""Print the phrase chosen for questions the development probe missed."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from phrase import candidates, inner_question
from run_durable import load_slices

needles = ("Từ Dụ",)
dev, _, _ = load_slices("2wiki", 120, 4)
for item in dev:
    if not any(needle.lower() in item["question"].lower() for needle in needles):
        continue
    print("\nQ", item["question"])
    print(" titles", [doc["title"] for doc in item["docs"] if doc.get("support")])
    rows = candidates(item["question"], item["docs"])
    if not rows:
        print("  no candidate")
        continue
    for row in rows[:4]:
        print(
            "  page", row["page"],
            "head", row["head"],
            "phrase", row["phrase"],
            "inner", inner_question(row["page"], row["phrase"], row["head"]),
        )
