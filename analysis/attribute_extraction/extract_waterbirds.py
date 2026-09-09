"""DeepSeek attribute extraction for Waterbirds traces (Table 8 prompt).

  buckets : bio (foreground bird) / env (background, habitat)
  input   : JSONL rows {answer}
  output  : + bio_attributes, env_attributes, bio_count, env_count
  env     : DEEPSEEK_API_KEY
"""
import os
import re
import json
import argparse
from typing import Any, Dict

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# ── model ──────────────────────────────────────────────────────────────────
MODEL_ID = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SYSTEM_PROMPT = """
You are an expert in analyzing a model's chain-of-thought.
Your job is to pull out the concrete evidence words or phrases the model itself cites and sort them into two buckets:

bio attribute: many morphological part, taxonomic label, features, adaptations or size/shape descriptor of the foreground object (e.g. wings, webbed feet, long legs, body shape, long tail, petrel).
env attribute: physical background or habitat terms that locate the scene (e.g. forest path, reeds, lake, ocean, coastal zone, sky, sand).

Respond strictly in this JSON format:

{
"bio_attributes":  [ ... ],
"env_attributes":  [ ... ],
"bio_count":       <integer>,
"env_count":       <integer>
}

Rules for the exracting attributes:
- A multi-word phrase like “long neck” counts as one attribute.
- Do not invent attributes; use only words or phrases literally present in the model output.

Example Input1: "The bird has a thick body, similar to a juvenile albatross, which are seabirds adapted to marine environments. They spend most of their time at sea and rely on oceanic ecosystems."
Output: {"bio_attributes": ["thick body", "juvenile albatross", "seabirds", "adapted to marine environments"], "env_attributes": ["sea", "oceanic ecosystems"], "bio_count": 4, "env_count": 2}

Example Input2: "The image shows a small animal with a light-colored face, dark eyes, and a body that's mostly light brown or beige. It has a small head with pointed ears, and its front paws are visible."
Output: {"bio_attributes": ["light-colored face", "dark eyes", "light brown", "beige", "small head", "pointed ears", "front paws"], "env_attributes": [], "bio_count": 7, "env_count": 0}

Example Input3: "The background has a body of water, like a pond or lake, and the bird is near that. Also, waterbirds often have adaptations for aquatic life, like webbed feet (though here it's a statue, but the context). The setting with water suggests it's a waterbird."
Output: {"bio_attributes": ["adaptations for aquatic life", "webbed feet", "waterbird"], "env_attributes": ["background", "body of water", "pond", "lake", "water"], "bio_count": 3, "env_count": 5}

Do not include any additional text or explanations. Your response should only contain the JSON object.
""".strip()

# ── helpers ────────────────────────────────────────────────────────────────
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.S)
    s = re.sub(r"\s*```$", "", s, flags=re.S)
    return s.strip()

def _def_result() -> Dict[str, Any]:
    return {
        "bio_attributes": [],
        "env_attributes": [],
        "bio_count": 0,
        "env_count": 0,
    }

def _extract_first_json_block(s: str) -> Dict[str, Any]:
    s = _strip_code_fences(s)
    start = s.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object start", s, 0)
    depth = 0
    end_idx = -1
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx == -1:
        raise json.JSONDecodeError("Unbalanced braces", s, start)
    return json.loads(s[start:end_idx])

# ── DeepSeek client ────────────────────────────────────────────────────────
def _new_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("set the DEEPSEEK_API_KEY environment variable")
    return OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def generate_chat(messages: list[Dict[str, str]],
                  max_new_tokens: int = 1024,
                  temperature: float = 0.0) -> str:
    client = _new_client()
    # DeepSeek exposes an OpenAI-compatible chat.completions endpoint
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=max(temperature, 1e-6),
        max_tokens=max_new_tokens,
        stream=False,
    )
    return resp.choices[0].message.content or ""

# ── extraction ─────────────────────────────────────────────────────────────
def extract_with_deepseek(answer_str: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Raw answer:\n{answer_str}"},
    ]
    reply_text = ""
    try:
        reply_text = generate_chat(messages, max_new_tokens=1024, temperature=0.0)
        parsed = _extract_first_json_block(reply_text)
        return parsed
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        res = _def_result()
        res["raw_reply"] = reply_text
        res["error"] = {
            "type": type(e).__name__,
            "msg": str(e),
            "raw_reply_head": (reply_text or "")[:200],
        }
        return res
    except (APIError, RateLimitError, APITimeoutError, Exception) as e:
        res = _def_result()
        res["raw_reply"] = reply_text
        res["error"] = {
            "type": type(e).__name__,
            "msg": str(e),
            "raw_reply_head": (reply_text or "")[:200],
        }
        return res

# ── conversion ─────────────────────────────────────────────────────────────
def convert(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8", buffering=1) as fout:
        for line_idx, line in enumerate(fin, start=1):
            obj = json.loads(line)
            raw = obj.get("answer", "")
            gpt_obj = extract_with_deepseek(raw)

            obj["bio_count"] = gpt_obj.get("bio_count", 0)
            obj["bio_attributes"] = gpt_obj.get("bio_attributes", [])
            obj["env_count"] = gpt_obj.get("env_count", 0)
            obj["env_attributes"] = gpt_obj.get("env_attributes", [])

            if "error" in gpt_obj:
                err = gpt_obj["error"] or {}
                obj["error_type"] = err.get("type")
                obj["error_msg"] = err.get("msg")
                obj["error_raw_reply_head"] = err.get("raw_reply_head")
                obj["error_line_index"] = line_idx
                obj["error_input_file"] = input_file
                if "id" in obj:
                    obj["error_obj_id"] = obj["id"]
                obj["error_model_id"] = MODEL_ID
                obj["error_prompt_tag"] = "v1"

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ── entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract bio/env attributes from Waterbirds reasoning traces (DeepSeek)")
    parser.add_argument("--input", required=True, help="input JSONL file")
    parser.add_argument("--output", required=True, help="output JSONL file")
    args = parser.parse_args()

    convert(args.input, args.output)
    print(f"done: {args.output}")
