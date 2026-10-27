import json

rows = []
for line in open("results/subst_live.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    if row["dataset"] == "hotpot":
        rows.append(row)
for row in rows:
    trace = row["trace"][0] if row["trace"] else {}
    print("Q", row["question"])
    print("  gold", row["answer"])
    print("  pred", row["preds"]["trustmem"])
    print("  decision", trace.get("decision"), "nmsg", len(trace.get("messages") or []), "composer", trace.get("composer"))
    print()
