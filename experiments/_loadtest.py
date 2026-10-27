import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM

name = os.environ.get("TRUSTMEM_LLM", "Qwen/Qwen2.5-3B-Instruct")
device = os.environ.get("TRUSTMEM_DEVICE", "cuda:0")
print("load", name, device, flush=True)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
print("ok", round(torch.cuda.memory_allocated(device) / 2**30, 2), "GiB", flush=True)
