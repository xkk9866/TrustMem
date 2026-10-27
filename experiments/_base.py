import json

rows = [json.loads(line) for line in open("results/subst_live.jsonl", encoding="utf-8") if line.strip()]
print("n", len(rows))
for row in rows:
    preds = row["preds"]
    print(row["answer"][:40])
    for name in ("trustmem", "joint", "ircot", "bm25", "selfask", "react", "direct", "open", "carry"):
        print(f"  {name:10} {preds.get(name, '')!r}"[:140])
    print()
