"""Check the engine: generation, logprobs, candidate scoring, throughput."""
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import wait_for_gpu  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen2.5-7B-Instruct-AWQ")
wait_for_gpu(0, 21000, 1500)
from engine import Engine  # noqa: E402
from common import SPECIALIST, render_pages, load, PLANNER  # noqa: E402

t = time.time()
eng = Engine(model, Path(__file__).resolve().parent / "results" / "cache_smoke.jsonl")
print("load", round(time.time() - t, 1), "s", flush=True)
docs = [{"title": "Summer's Blood", "text": "Summer's Blood is a 2009 horror film directed by Lee Demarbre."},
        {"title": "Lee Demarbre", "text": "Lee Demarbre (born in Chicoutimi, Quebec) is a Canadian film director."}]
user = SPECIALIST.format(pages=render_pages(docs[:1]), question="Who directed Summer's Blood?")
print(eng.run([("gen", user, 32, ("\n",), 0.0, 0)]))
lps = eng.run([("score", user, ("Lee Demarbre", "Marceline Duval", "NONE", "Chicoutimi"))])[0]
z = max(lps); ps = [math.exp(x - z) for x in lps]; s = sum(ps)
print("scores", [round(x, 3) for x in lps], "probs", [round(p / s, 3) for p in ps])
user2 = SPECIALIST.format(pages=render_pages(docs[:1]), question="Where was Lee Demarbre born?")
print(eng.run([("gen", user2, 32, ("\n",), 0.0, 0)]), eng.run([("score", user2, ("Chicoutimi", "NONE"))]))
items = load("hotpot", 64, 900)
t = time.time()
plans = eng.run([("gen", PLANNER.format(question=it["question"]), 220, ("\n\nQuestion", "\nQuestion:"), 0.0, 0) for it in items])
print("64 plans", round(time.time() - t, 1), "s")
for it, (p, lp) in list(zip(items, plans))[:6]:
    print("Q", it["question"], "\n", p, round(lp, 2))
reqs = [("gen", SPECIALIST.format(pages=render_pages(it["docs"][:5]), question=it["question"]), 32, ("\n",), 0.0, 0) for it in items]
t = time.time()
outs = eng.run(reqs)
print("64 specialist gens", round(time.time() - t, 1), "s")
sreqs = [("score", r[1], (o[0].strip() or "x", "Paris", "NONE")) for r, o in zip(reqs, outs)]
t = time.time()
eng.run(sreqs)
print("64 scores x3", round(time.time() - t, 1), "s", eng.stats)
