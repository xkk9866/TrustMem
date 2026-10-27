"""One frozen, int4-quantised LLM behind a request cache.

Two request kinds:
  ("gen", user, max_tokens, stop, temperature, seed)  -> (text, logprob of the text)
  ("score", user, candidates)                       -> [log P(candidate + end-of-turn | user)]
  ("choice", user, options)                         -> [log P(first token of option | user)]

Scores are computed as products of next-token probabilities.  Each needed context
(prompt + candidate prefix) is sent as a one-token request with top-20 logprobs,
so vLLM's prefix cache serves the shared page prompt.  A token outside the top 20
is charged the 20th logprob, an upper bound on its probability.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

TOPK = 20


def _key(req: tuple) -> str:
    return hashlib.sha1(json.dumps(req, ensure_ascii=False).encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, object] = {}
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        key, value = json.loads(line)
                    except Exception:
                        continue
                    self.data[key] = value
        self.handle = open(path, "a", encoding="utf-8")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.handle.write(json.dumps([key, value], ensure_ascii=False) + "\n")

    def flush(self):
        self.handle.flush()


class Engine:
    def __init__(self, model: str, cache_path: Path, max_model_len: int = 8192, gpu_util: float = 0.85):
        from vllm import LLM

        self.model_name = model
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
            enable_prefix_caching=True,
            max_logprobs=TOPK,
            seed=0,
            dtype="float16",
        )
        self.tok = self.llm.get_tokenizer()
        self.qwen3 = "qwen3" in model.lower()
        self.end_text = "<|im_end|>" if "qwen" in model.lower() else self.tok.eos_token
        self.end_id = self.tok.convert_tokens_to_ids(self.end_text)
        self.cache = Cache(cache_path)
        self.stats = {"gen": 0, "score_ctx": 0, "hit": 0, "prompt_tokens": 0}
        self._ids: dict[str, list[int]] = {}

    # ------------------------------------------------------------ prompts
    def render(self, user: str) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.qwen3:
            kwargs["enable_thinking"] = False
        return self.tok.apply_chat_template([{"role": "user", "content": user}], **kwargs)

    def ids(self, user: str) -> list[int]:
        got = self._ids.get(user)
        if got is None:
            got = self.tok(self.render(user), add_special_tokens=False).input_ids
            if len(self._ids) > 200000:
                self._ids.clear()
            self._ids[user] = got
        return got

    def cont_ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids + [self.end_id]

    # ------------------------------------------------------------ run
    def run(self, reqs: list[tuple]) -> list:
        out = [None] * len(reqs)
        gen_todo, score_todo, choice_todo = [], [], []
        for index, req in enumerate(reqs):
            hit = self.cache.get(_key(req))
            if hit is not None:
                out[index] = hit
                self.stats["hit"] += 1
            elif req[0] == "gen":
                gen_todo.append(index)
            elif req[0] == "choice":
                choice_todo.append(index)
            else:
                score_todo.append(index)
        if gen_todo:
            self._run_gen(reqs, gen_todo, out)
        if choice_todo:
            self._run_choice(reqs, choice_todo, out)
        if score_todo:
            self._run_score(reqs, score_todo, out)
        self.cache.flush()
        return out

    def _run_gen(self, reqs, todo, out):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        unique = {}
        for index in todo:
            unique.setdefault(_key(reqs[index]), []).append(index)
        keys = list(unique)
        prompts, params = [], []
        for key in keys:
            _, user, max_tokens, stop, temperature, seed = reqs[unique[key][0]]
            ids = self.ids(user)
            self.stats["prompt_tokens"] += len(ids)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(
                SamplingParams(
                    temperature=temperature,
                    top_p=1.0,
                    max_tokens=max_tokens,
                    stop=list(stop) if stop else None,
                    seed=seed if temperature > 0 else None,
                    logprobs=0,
                )
            )
        results = self.llm.generate(prompts, params, use_tqdm=False)
        for key, res in zip(keys, results):
            comp = res.outputs[0]
            lp = comp.cumulative_logprob if comp.cumulative_logprob is not None else -50.0
            value = [comp.text, float(lp)]
            self.cache.put(key, value)
            for index in unique[key]:
                out[index] = value
        self.stats["gen"] += len(keys)

    def _run_choice(self, reqs, todo, out):
        """One forward step per prompt; read the logprob of each option's first token."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        unique = {}
        for index in todo:
            unique.setdefault(_key(reqs[index]), []).append(index)
        keys = list(unique)
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=TOPK)
        prompts = [TokensPrompt(prompt_token_ids=self.ids(reqs[unique[k][0]][1])) for k in keys]
        results = self.llm.generate(prompts, params, use_tqdm=False)
        for key, res in zip(keys, results):
            table = {tid: float(lp.logprob) for tid, lp in res.outputs[0].logprobs[0].items()}
            floor = min(table.values())
            options = reqs[unique[key][0]][2]
            value = []
            for opt in options:
                tid = self.tok(opt, add_special_tokens=False).input_ids[0]
                value.append(table.get(tid, floor))
            self.cache.put(key, value)
            for index in unique[key]:
                out[index] = value
        self.stats["choice"] = self.stats.get("choice", 0) + len(keys)

    def _run_score(self, reqs, todo, out):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        contexts: dict[tuple, None] = {}
        plans = {}
        for index in todo:
            _, user, cands = reqs[index]
            base = self.ids(user)
            per = []
            for cand in cands:
                cont = self.cont_ids(cand)
                steps = []
                for t in range(len(cont)):
                    ctx = tuple(base + cont[:t])
                    contexts.setdefault(ctx, None)
                    steps.append((ctx, cont[t]))
                per.append(steps)
            plans[index] = per
        ctx_list = list(contexts)
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=TOPK)
        # Shorter contexts first so that the page prefix enters the cache early.
        order = sorted(range(len(ctx_list)), key=lambda i: len(ctx_list[i]))
        dist: dict[tuple, dict[int, float]] = {}
        chunk = 4096
        for start in range(0, len(order), chunk):
            part = [ctx_list[i] for i in order[start : start + chunk]]
            results = self.llm.generate(
                [TokensPrompt(prompt_token_ids=list(ctx)) for ctx in part], params, use_tqdm=False
            )
            for ctx, res in zip(part, results):
                lps = res.outputs[0].logprobs[0]
                dist[ctx] = {tid: float(lp.logprob) for tid, lp in lps.items()}
        self.stats["score_ctx"] += len(ctx_list)
        for index, per in plans.items():
            values = []
            for steps in per:
                total = 0.0
                for ctx, tid in steps:
                    table = dist[ctx]
                    total += table.get(tid, min(table.values()))
                values.append(total)
            self.cache.put(_key(reqs[index]), values)
            out[index] = values


class MockEngine:
    """Deterministic stand-in used to test control flow on a CPU."""

    def __init__(self, cache_path: Path | None = None):
        self.stats = {"gen": 0, "score_ctx": 0, "hit": 0, "prompt_tokens": 0}
        self.model_name = "mock"

    def run(self, reqs):
        import re

        out = []
        for req in reqs:
            if req[0] in ("score", "choice"):
                out.append([-1.0 - 0.5 * i - len(c) * 0.01 for i, c in enumerate(req[2])])
                continue
            user = req[1]
            self.stats["gen"] += 1
            if user.startswith("Break the question"):
                q = user.rsplit("Question:", 1)[1].strip()
                out.append([f"#1 = LOOKUP: Who is related to {q[:30]}?\n#2 = LOOKUP: Where was #1 born?\nANSWER: #2", -3.0])
            elif "specialist agent" in user:
                names = re.findall(r"## ([^\n]+)", user)
                out.append([names[0] if names else "NONE", -1.0])
            elif "plausible but is factually wrong" in user:
                out.append(["Marceline Duval", -0.5])
            else:
                out.append(["Paris", -1.0])
        return out


def sanity(engine) -> None:
    """Print one generation and one candidate scoring so the log shows the engine is sane."""
    import math as _m

    from common import SPECIALIST, render_pages

    docs = [{"title": "Summer's Blood", "text": "Summer's Blood is a 2009 horror film directed by Lee Demarbre."}]
    user = SPECIALIST.format(pages=render_pages(docs), question="Who directed Summer's Blood?")
    t = time.time()
    print("sanity gen", engine.run([("gen", user, 32, ("\n",), 0.0, 0)]), flush=True)
    lps = engine.run([("score", user, ("Lee Demarbre", "Marceline Duval", "NONE"))])[0]
    z = max(lps)
    ps = [_m.exp(x - z) for x in lps]
    print("sanity score", [round(x, 3) for x in lps], [round(p / sum(ps), 4) for p in ps], f"{time.time() - t:.2f}s", flush=True)
    from common import VERIFY

    for value in ("Lee Demarbre", "Marceline Duval", "2009"):
        lp = engine.run([("choice", VERIFY.format(pages=render_pages(docs), question="Who directed Summer's Blood?", value=value), ("Yes", "No"))])[0]
        print("sanity verify", value, [round(x, 3) for x in lp], round(1 / (1 + _m.exp(lp[1] - lp[0])), 4), flush=True)
