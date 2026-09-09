"""Loaders and prompt builders for the four evaluated models.

  qwen3-vl-8b-thinking   Qwen/Qwen3-VL-8B-Thinking
  glm                    zai-org/GLM-4.1V-9B-Thinking
  intern-s1              internlm/Intern-S1-mini
  r1-onevision           Fancy-MLLM/R1-Onevision-7B-RL

  select_key(path) : substring match on the model path
  ATTN_IMPL=eager  : force eager attention (analysis/attention_blocking)
"""
import os

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Glm4vForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

__all__ = ["LOADERS", "BUILDERS", "select_key"]

_ATTN_IMPL = os.environ.get("ATTN_IMPL", "").strip() or None
_ATTN_KW = {"attn_implementation": _ATTN_IMPL} if _ATTN_IMPL else {}


# ── loaders ────────────────────────────────────────────────────────────────
def load_qwen3_thinking(path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", **_ATTN_KW
    )
    proc = AutoProcessor.from_pretrained(path)
    return model, proc


def load_glm(path):
    model = Glm4vForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", **_ATTN_KW
    )
    proc = AutoProcessor.from_pretrained(path)
    return model, proc


def load_intern(path):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, **_ATTN_KW
    )
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    return model, proc


def load_r1_onevision(path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", **_ATTN_KW
    )
    proc = AutoProcessor.from_pretrained(path)
    return model, proc


# ── prompt builders ────────────────────────────────────────────────────────
def _messages(image, question):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]


def build_chat_prompt(proc, image, question):
    """Default chat-template prompt (Qwen3-VL-Thinking, GLM-4.1V-Thinking, R1-OneVision)."""
    messages = _messages(image, question)
    txt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ims, vids = process_vision_info(messages)
    return {"text": [txt], "images": ims, "videos": vids}


def build_intern_thinking_prompt(proc, image, question):
    """Intern-S1 needs ``enable_thinking=True`` to emit a reasoning trace."""
    messages = _messages(image, question)
    txt = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    ims, vids = process_vision_info(messages)
    return {"text": [txt], "images": ims, "videos": vids}


# ── registry ───────────────────────────────────────────────────────────────
LOADERS = {
    "qwen3-vl-8b-thinking": load_qwen3_thinking,
    "glm":                  load_glm,
    "intern-s1":            load_intern,
    "r1-onevision":         load_r1_onevision,
}

BUILDERS = {
    "qwen3-vl-8b-thinking": build_chat_prompt,
    "glm":                  build_chat_prompt,
    "intern-s1":            build_intern_thinking_prompt,
    "r1-onevision":         build_chat_prompt,
}


def select_key(path: str) -> str:
    path = path.lower()
    for k in LOADERS:
        if k in path:
            return k
    raise ValueError(f"unsupported model: {path} (supported: {list(LOADERS)})")
