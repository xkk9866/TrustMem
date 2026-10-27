"""Dump the recovered open-retrieval functions so the method can be rebuilt."""

import dis
import io
import marshal
import os
import sys
import types
from pathlib import Path

path = Path("__pycache__/run_openrag.cpython-313.pyc")
code = marshal.loads(path.read_bytes()[16:])
module = types.ModuleType("run_openrag")
module.__file__ = str(Path("run_openrag.py").resolve())
sys.modules["run_openrag"] = module
exec(code, module.__dict__)

names = [
    "answer_method",
    "attested_titles",
    "bm25_retrieve",
    "dense_retrieve",
    "crag_rows",
    "rrf_fuse",
    "first_hop_lists",
    "second_hop",
    "place_new_page",
    "retrieve",
    "read_store",
    "pack_answer",
    "horizon_for",
    "evict",
    "diversify",
    "gold_rows",
    "support_recall",
    "surface_entities",
    "page_sentence",
    "phrase_in",
    "query_text",
    "agent_specs",
    "rows_of",
    "embed_matrix",
]
out = Path("_openrag_dis.txt")
buf = io.StringIO()
for name in names:
    fn = getattr(module, name)
    buf.write("\n" + "=" * 20 + " " + name + " " + "=" * 20 + "\n")
    buf.write(f"args {fn.__code__.co_varnames[:fn.__code__.co_argcount]}\n")
    buf.write(f"vars {fn.__code__.co_varnames}\n")
    buf.write(f"names {fn.__code__.co_names}\n")
    consts = [c for c in fn.__code__.co_consts if not hasattr(c, "co_code")]
    buf.write(f"consts {consts}\n")
    dis.dis(fn, file=buf)
out.write_text(buf.getvalue(), encoding="utf-8")
print("wrote", out, "bytes", out.stat().st_size)
os._exit(0)
