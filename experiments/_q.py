import json

rows = [json.loads(line) for line in open("results/subst_live.jsonl", encoding="utf-8") if line.strip()]
for row in rows:
    if row["answer"] not in ("Marc Allégret", "Gülçiçek Hatun"):
        continue
    print("Q", row["question"])
    print("gold", row["answer"], "pred", row["preds"]["trustmem"])
    print(json.dumps(row["trace"], ensure_ascii=False)[:1200])
    print("---")
