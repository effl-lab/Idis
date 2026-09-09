"""DeepSeek attribute extraction for Idis-perception traces (Table 6 prompt).

  categories : dog, bird, vehicle, reptile, carnivore, insect, instrument, primate, fish, other
  input      : JSONL rows {answer}; filename <model>-<class>-<n>-<case>.jsonl
  output     : <out_prefix>.r{k}.out.jsonl  (+ .errors.jsonl, retried with --loop)
  env        : DEEPSEEK_API_KEY
"""
import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

# ── model ──────────────────────────────────────────────────────────────────
MODEL_ID = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
RETRY_BACKOFFS = [1.0, 2.0, 4.0, 8.0]

# ── class → representative distractor objects (Table 3) ────────────────────
OBJECTS: Dict[str, List[str]] = {
    "dog": ["dog bone chew toy", "dog bowl", "tennis ball", "kennel"],
    "bird": ["birdcage", "nest", "feather", "bird feeder"],
    "wheeled vehicle": ["tire", "steering wheel", "license plate", "bumper"],
    "reptile": ["terrarium rock", "heat lamp", "log hideout", "shed skin"],
    "carnivore": ["fang", "blood stain", "chunk of meat", "skeletal animal carcass"],
    "insect": ["trash bag", "empty net designed for catching insects", "fruit peel", "flower"],
    "musical instrument": ["chalkboard with music notes", "music stand", "metronome", "sheet music"],
    "primate": ["patch of jungle foliage", "banana", "coconut", "vine"],
    "fish": ["fishing rod", "large empty nylon fishing net", "life jacket", "aquarium coral ornament"],
}

LABEL_ALIASES = {
    "vehicle": "wheeled vehicle",
    "instrument": "musical instrument",
}

ATTR_KEYS = [
    "dog_attributes", "bird_attributes", "vehicle_attributes",
    "reptile_attributes", "carnivore_attributes", "insect_attributes",
    "instrument_attributes", "primate_attributes", "fish_attributes",
    "other_attributes",
]
ERROR_KEYS = ["error_type", "error_msg", "error_raw_reply", "error_line_index",
              "error_input_file", "error_model_id", "error_prompt_tag"]

# ── system prompt (Table 6) ────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert in analyzing a model's chain-of-thought. 
Extract literal evidence words or phrases from the text and classify them into 10 categories: nine main classes (dog, bird, vehicle, reptile, carnivore, insect, instrument, primate, fish) and one “other” category for anything else. 
For each main class, include attributes or objects directly related to it, considering morphology, taxonomy, features, shape, size, or adaptations. 

Representative related objects:
- dog_attributes: dog bone chew toy, dog bowl, tennis ball, kennel 
- bird_attributes: birdcage, nest, feather, bird feeder 
- vehicle_attributes: tire, steering wheel, license plate, bumper 
- reptile_attributes: terrarium rock, heat lamp, log hideout, shed skin 
- carnivore_attributes: fang, blood stain, chunk of meat, skeletal animal carcass 
- insect_attributes: trash bag, insect net, fruit peel, flower 
- instrument_attributes: chalkboard with music notes, music stand, metronome, sheet music 
- primate_attributes: jungle foliage, banana, coconut, vine 
- fish_attributes: fishing rod, fishing net, life jacket, aquarium coral ornament 
- other_attributes: unrelated attributes or objects (e.g., umbrella, clock, tv, suitcase, etc.)

Rules: 
- Use only literal words/phrases from the text (case-insensitive match for listed objects). 
- Multi-word phrases (e.g., “long tail”) count as one attribute. 
- Do not infer or paraphrase. 
- “Taxonomic labels” like “bird”, “dog”, etc. are valid only if they literally appear. 
- Each extracted attribute must belong to exactly one of the 10 categories.

Respond strictly in this JSON format:
{
  "dog_attributes": [...],
  "bird_attributes": [...],
  "vehicle_attributes": [...],
  "reptile_attributes": [...],
  "carnivore_attributes": [...],
  "insect_attributes": [...],
  "instrument_attributes": [...],
  "primate_attributes": [...],
  "fish_attributes": [...],
  "other_attributes": [...],
  "counts": { "dog": <int>, "bird": <int>, "vehicle": <int>, "reptile": <int>, "carnivore": <int>, "insect": <int>, "instrument": <int>, "primate": <int>, "fish": <int>, "other": <int>}
}

Only output the JSON object. No explanations or extra text.
""".strip()


# ── helpers ────────────────────────────────────────────────────────────────
CODE_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", flags=re.I)
CODE_FENCE_CLOSE_RE = re.compile(r"\s*```$", flags=re.I)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = CODE_FENCE_OPEN_RE.sub("", s)
    s = CODE_FENCE_CLOSE_RE.sub("", s)
    return s.strip()


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


def _normalize_label(s: str) -> str:
    s = (s or "").strip().replace("_", " ").lower()
    return LABEL_ALIASES.get(s, s)


def parse_label_and_k_from_filename(path: str):
    """<model>-<class>-<n>-<case>[.partNNN][.rK.out|errors].jsonl → (class, n)."""
    stem, _ = os.path.splitext(os.path.basename(path))
    stem = re.sub(r"\.r\d+\.(out|errors)$", "", stem)
    stem = re.sub(r"\.part\d+$", "", stem)
    parts = stem.split("-")
    if len(parts) < 2:
        return None, None
    label = _normalize_label(parts[1].split(".", 1)[0])
    k = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return label, k


def _def_result() -> Dict[str, Any]:
    out = {k: [] for k in ATTR_KEYS}
    out["counts"] = {c: 0 for c in ["dog", "bird", "vehicle", "reptile", "carnivore",
                                    "insect", "instrument", "primate", "fish", "other"]}
    return out


# ── DeepSeek client ────────────────────────────────────────────────────────
def _new_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("set the DEEPSEEK_API_KEY environment variable")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def generate_chat(system_prompt: str, user_content: str) -> str:
    client = _new_client()
    last_err: Optional[Exception] = None
    for backoff in [0.0] + RETRY_BACKOFFS:
        if backoff > 0:
            time.sleep(backoff)
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=1e-6,
                max_tokens=512,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            break
    raise last_err if last_err else RuntimeError("Unknown API error")


# ── extraction ─────────────────────────────────────────────────────────────
def extract_with_deepseek(answer_text: str) -> Dict[str, Any]:
    reply_text = ""
    try:
        reply_text = generate_chat(SYSTEM_PROMPT, f"Raw text:\n{answer_text}")
        parsed = _extract_first_json_block(reply_text)
        out = _def_result()
        for key in ATTR_KEYS:
            out[key] = list(parsed.get(key, []) or [])
        counts = parsed.get("counts") or {}
        for c in out["counts"]:
            out["counts"][c] = int(counts.get(c, len(out[f"{c}_attributes"])))
        return out
    except Exception as e:
        res = _def_result()
        res["error"] = {"type": type(e).__name__, "msg": str(e), "raw_reply": (reply_text or "")[:200]}
        return res


def convert(input_file: str, output_file: str) -> None:
    """One extraction pass over input_file → output_file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    file_label, _ = parse_label_and_k_from_filename(input_file)
    if not file_label:
        raise ValueError(f"cannot parse the target class from the filename: {input_file}")

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8", buffering=1) as fout:
        for idx, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            result = extract_with_deepseek(obj.get("answer", ""))

            for key in ATTR_KEYS:
                obj[key] = result.get(key, [])
            obj["counts"] = result.get("counts", {})

            if "error" in result:
                err = result["error"] or {}
                obj["error_type"] = err.get("type")
                obj["error_msg"] = err.get("msg")
                obj["error_raw_reply"] = err.get("raw_reply")
                obj["error_line_index"] = idx
                obj["error_input_file"] = input_file
                obj["error_model_id"] = MODEL_ID
                obj["error_prompt_tag"] = "ten_category_v1_answer_full"
            else:
                for k in ERROR_KEYS:
                    obj.pop(k, None)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"done: {output_file}")


def make_error_subset(recent_out_file: str, next_retry_input_file: str) -> int:
    """Write the rows that failed in recent_out_file (error_msg set) as the next retry input."""
    count = 0
    with open(recent_out_file, "r", encoding="utf-8") as fin, \
         open(next_retry_input_file, "w", encoding="utf-8", buffering=1) as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("error_msg"):
                for k in ATTR_KEYS + ["counts"] + ERROR_KEYS:
                    obj.pop(k, None)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
    print(f"{count} rows failed → {next_retry_input_file}" if count else "no errors in this round")
    return count


def loop_convert_until_clean(first_input: str, out_prefix: str, max_rounds: int = 100,
                             sleep_between_sec: float = 0.0) -> None:
    """Round 0 processes first_input; every later round retries only the failed rows."""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    cur_input = first_input
    for r in range(max_rounds):
        out_file = f"{out_prefix}.r{r}.out.jsonl"
        err_file = f"{out_prefix}.r{r}.errors.jsonl"
        convert(cur_input, out_file)
        if make_error_subset(out_file, err_file) == 0:
            print(f"finished after {r + 1} round(s); last output: {out_file}")
            return
        cur_input = err_file
        if sleep_between_sec > 0:
            time.sleep(sleep_between_sec)
    print(f"reached max_rounds={max_rounds}; remaining errors in {cur_input}")


# ── entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="input JSONL (filename encodes the target class)")
    parser.add_argument("--out_prefix", required=True, help="output prefix; .r{k}.out.jsonl / .r{k}.errors.jsonl are appended")
    parser.add_argument("--loop", action="store_true", help="retry failed rows until none remain (bounded by --max_rounds)")
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--sleep_between", type=float, default=0.0, help="seconds to wait between rounds (rate limiting)")
    args = parser.parse_args()

    if args.loop:
        loop_convert_until_clean(args.input, args.out_prefix, max_rounds=args.max_rounds,
                                 sleep_between_sec=args.sleep_between)
    else:
        out_file = f"{args.out_prefix}.r0.out.jsonl"
        convert(args.input, out_file)
        make_error_subset(out_file, f"{args.out_prefix}.r0.errors.jsonl")
