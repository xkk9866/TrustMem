"""Run methods x conditions x datasets with one engine, batching across questions.

One-off:
  python run.py --datasets hotpot,2wiki --n 500 --methods trustmem,self_ask --conditions clean,poison1 --tag main
Server (holds the GPU and executes spec files dropped into queue/):
  python run.py --serve --model ~/models/Qwen2.5-7B-Instruct-AWQ
A spec file is JSON with the same fields as the command line options.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib  # noqa: E402

import common as C  # noqa: E402
import methods as M  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
QUEUE = ROOT / "queue"

CONDITIONS = {
    "clean": (0, False),
    "poison1": (1, False),
    "poison5": (5, False),
    "adapt1": (1, True),
    "adapt5": (5, True),
    "target1": (1, "targeted"),
    "target5": (5, "targeted"),
}


def wait_for_gpu(index: int, need_free_mib: int, max_used_mib: int) -> None:
    """Block until nobody else is using the GPU."""
    import subprocess

    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
            )
            used, total, util = [int(x) for x in out.strip().split(",")]
        except Exception as exc:  # pragma: no cover
            print("nvidia-smi failed", exc, flush=True)
            used, total, util = 0, 10**6, 0
        if used <= max_used_mib and total - used >= need_free_mib:
            print(time.strftime("%H:%M:%S"), f"gpu {index} free: used {used} MiB util {util}%", flush=True)
            return
        print(time.strftime("%H:%M:%S"), f"gpu {index} busy: used {used} MiB util {util}%; waiting", flush=True)
        time.sleep(120)


class Job:
    def __init__(self, key, dataset, method, cond, item, ctx, gen):
        self.key, self.dataset, self.method, self.cond, self.item, self.ctx, self.gen = key, dataset, method, cond, item, ctx, gen
        self.pending = None
        self.n_gen = 0
        self.n_score = 0
        self.rounds = 0


def run_spec(engine, spec: dict, model_tag: str) -> None:
    datasets = spec.get("datasets", "hotpot,2wiki,musique").split(",")
    n, offset = int(spec.get("n", 500)), int(spec.get("offset", 0))
    methods = spec.get("methods", "trustmem").split(",")
    conds = spec.get("conditions", "clean").split(",")
    k, seed = int(spec.get("k", 2)), int(spec.get("seed", 0))
    mirror = int(spec.get("mirror", 1))
    rtag = "" if mirror == 1 else f"|r{mirror}"
    pool = int(spec.get("pool", 192))
    tag = spec.get("tag", "main")

    RESULTS.mkdir(exist_ok=True)
    importlib.reload(C)
    importlib.reload(M)
    load, partition, score_item, f1 = C.load, C.partition, C.score_item, C.f1
    METHODS, Ctx = M.METHODS, M.Ctx
    conditions = getattr(M, "CONDITIONS", None) or {c: dict(poison=v[0], adaptive=v[1] is True, targeted=v[1] == "targeted")
                                                    for c, v in CONDITIONS.items()}
    out_path = RESULTS / f"{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                continue
    todo = []
    for dataset in datasets:
        items = load(dataset, n, offset)
        for cond in conds:
            for method in methods:
                for item in items:
                    key = f"{model_tag}|{dataset}|{method}|{cond}|k{k}|s{seed}{rtag}|{item['id']}"
                    if key not in done:
                        todo.append((key, dataset, method, cond, item))
    print(time.strftime("%H:%M:%S"), f"[{tag}] jobs {len(todo)} (done {len(done)})", flush=True)

    out = open(out_path, "a", encoding="utf-8")
    active: list[Job] = []
    cursor = 0
    finished = 0
    t0 = time.time()
    last = t0

    def advance(job, resps):
        try:
            reqs = job.gen.send(resps) if resps is not None else next(job.gen)
            while not reqs:  # a generator may yield an empty batch
                reqs = job.gen.send([])
            job.pending = reqs
            job.rounds += 1
            for r in reqs:
                if r[0] == "gen":
                    job.n_gen += 1
                elif r[0] == "choice":
                    job.n_score += 1
                else:
                    job.n_score += len(r[2])
            return True
        except StopIteration as stop:
            finish(job, stop.value)
            job.pending = None
            return False

    def finish(job, pred):
        nonlocal finished
        sc = score_item(pred or "", job.item)
        pvals = job.ctx.log.get("poison_values", [])
        adv = max((f1(pred or "", v) for v in pvals), default=0.0)
        row = {
            "key": job.key, "model": model_tag, "dataset": job.dataset, "method": job.method,
            "cond": job.cond, "k": k, "seed": seed, "mirror": mirror, "id": job.item["id"], "type": job.item["type"],
            "question": job.item["question"], "answer": job.item["answer"], "pred": pred or "",
            "f1": sc["f1"], "em": sc["em"], "adv": adv, "n_gen": job.n_gen, "n_score": job.n_score,
            "rounds": job.rounds, "log": job.ctx.log,
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        finished += 1

    def start(entry):
        key, dataset, method, cond, item = entry
        fn, kwargs = METHODS[method]
        ctx = Ctx(item=item, groups=partition(item, k, seed, mirror), mirror=mirror, seed=seed, **conditions[cond])
        job = Job(key, dataset, method, cond, item, ctx, fn(ctx, **kwargs))
        advance(job, None)
        return job

    abort = QUEUE / f"ABORT_{tag}"
    while cursor < len(todo) or active:
        if abort.exists():
            abort.unlink()
            print(time.strftime("%H:%M:%S"), f"[{tag}] aborted", flush=True)
            break
        while cursor < len(todo) and len(active) < pool:
            job = start(todo[cursor])
            cursor += 1
            if job.pending is not None:
                active.append(job)
        if not active:
            continue
        flat, spans = [], []
        for job in active:
            spans.append((len(flat), len(job.pending)))
            flat.extend(job.pending)
        resps = engine.run(flat)
        still = []
        for job, (at, cnt) in zip(active, spans):
            if advance(job, resps[at : at + cnt]):
                still.append(job)
        active = still
        out.flush()
        now = time.time()
        if now - last > 120 or (cursor >= len(todo) and not active):
            rate = finished / max(1e-6, now - t0)
            print(time.strftime("%H:%M:%S"), f"[{tag}] done {finished}/{len(todo)} active {len(active)} rate {rate:.2f}/s", engine.stats, flush=True)
            last = now
    out.close()
    print(time.strftime("%H:%M:%S"), f"[{tag}] finished in {time.time() - t0:.0f}s", flush=True)


def serve(engine, model_tag: str, idle_hours: float) -> None:
    QUEUE.mkdir(exist_ok=True)
    (QUEUE / "done").mkdir(exist_ok=True)
    idle_since = time.time()
    while True:
        specs = sorted(p for p in QUEUE.glob("*.json"))
        specs = [p for p in specs if json.loads(p.read_text()).get("model_tag", model_tag) == model_tag]
        if not specs:
            if (QUEUE / "STOP").exists() or time.time() - idle_since > idle_hours * 3600:
                print(time.strftime("%H:%M:%S"), "queue empty; exiting", flush=True)
                return
            time.sleep(20)
            continue
        path = specs[0]
        spec = json.loads(path.read_text())
        print(time.strftime("%H:%M:%S"), "spec", path.name, spec, flush=True)
        try:
            run_spec(engine, spec, model_tag)
            path.rename(QUEUE / "done" / path.name)
        except Exception:
            import traceback

            traceback.print_exc()
            path.rename(QUEUE / "done" / (path.name + ".failed"))
        idle_since = time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hotpot,2wiki,musique")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--methods", default="trustmem")
    ap.add_argument("--conditions", default="clean")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mirror", type=int, default=1)
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-7B-Instruct-AWQ"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--pool", type=int, default=192)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--idle-hours", type=float, default=3.0)
    args = ap.parse_args()

    model_tag = "mock" if args.mock else Path(args.model).name
    if args.mock:
        from engine import MockEngine

        engine = MockEngine()
    else:
        if not args.no_wait:
            wait_for_gpu(args.gpu, need_free_mib=21000, max_used_mib=1500)
        from engine import Engine, sanity

        engine = Engine(args.model, RESULTS / f"cache_{model_tag}.jsonl", gpu_util=args.gpu_util)
        sanity(engine)
    if args.serve:
        serve(engine, model_tag, args.idle_hours)
        return
    run_spec(engine, vars(args), model_tag)


if __name__ == "__main__":
    main()
