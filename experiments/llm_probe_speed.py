import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

name = os.environ.get("TRUSTMEM_LLM", "Qwen/Qwen2.5-7B-Instruct")
device = os.environ.get("TRUSTMEM_DEVICE", "cuda:1")
print("transformers", transformers.__version__, "torch", torch.__version__)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device)
model.eval()
print("loaded", round(time.time() - t0, 1), "s", torch.cuda.memory_allocated(device) / 2**30, "GiB")
text = "Lee Demarbre is a Canadian film director. Born in Chicoutimi, Quebec, Demarbre graduated ... " * 40
msgs = [{"role": "user", "content": text + "\nWhere was Lee Demarbre born? Reply with a short span."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(prompt, return_tensors="pt").input_ids.to(device)
print("prompt tokens", ids.shape[1])
for _ in range(3):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=24, do_sample=False)
    print(round(time.time() - t0, 2), "s", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
