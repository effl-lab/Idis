#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DeepSeek attribute extraction for Idis-math traces (Table 7 prompt), one call per sentence.

  input  : JSONL rows {answer (sentence), line_no, sentence_index,
                       target_labels, distractor_labels, question}
  output : + target_related / distractor_related / other, counts
  env    : DEEPSEEK_API_KEY
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover
    OpenAI = None
    APIError = APITimeoutError = RateLimitError = Exception

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


MODEL_ID = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
RETRY_BACKOFFS = [1.0, 2.0, 4.0, 8.0]

SYSTEM_PROMPT = """
You are analyzing a model's mathematical visual reasoning trace for the
MathVerse conflicting setting, where a TARGET geometric figure (the figure the
original question is actually about) coexists with one or more DISTRACTOR
figures (extra shapes inserted into the image that are NOT the subject of the
question).

The user message provides four pieces of context for this sample:
  TARGET shapes:     <comma-separated shape names>
  DISTRACTOR shapes: <comma-separated shape names>
  QUESTION:          <the original problem statement, text-only version>
  Raw text:          a single sentence from the model's reasoning trace.

Use the QUESTION as the authoritative description of the TARGET. Any concrete
entity, point, line, segment, angle, length, value, relation, or option that
appears in the QUESTION is part of the TARGET context — when the SAME entity
(or a clear anaphoric reference to it) appears in the Raw text, classify the
matching phrase as target_related.

Task: Extract every concrete mathematical, diagrammatic, or visual attribute
phrase that literally appears in the Raw text, and group each phrase into
exactly one of these three categories:
  target_related, distractor_related, other.

Definitions:
- target_related: phrases that refer to (a) a TARGET shape, OR (b) any entity,
  property, measurement, relation, value, or option mentioned in the QUESTION,
  OR (c) a part/property/operation that the surrounding sentence clearly
  attaches to the TARGET (e.g. a vertex, side, angle, area, congruence claim,
  or theorem usage about the TARGET).
- distractor_related: phrases that refer to a DISTRACTOR shape or to a
  part/property/measurement/relation/operation explicitly tied to a
  DISTRACTOR shape in the Raw text.
- other: concrete attribute phrases that cannot be confidently assigned to
  either the TARGET or the DISTRACTOR — generic math terminology, unrelated
  entities, or otherwise ambiguous mentions.

Rules:
- Use only words/phrases that literally appear in the Raw text. Do not infer,
  translate, or paraphrase. The QUESTION is for grounding only — do not
  extract from it.
- A phrase counts as target_related if it names or clearly refers to anything
  introduced by the QUESTION or to the TARGET shape. Anaphora ("it", "the
  triangle") counts only when the referent is unambiguous from the sentence.
- A phrase counts as distractor_related only when the Raw text makes the link
  to a DISTRACTOR shape explicit (e.g. names the distractor shape, describes
  a property of it, or compares it against the TARGET).
- If the link is ambiguous, put the phrase in "other".
- Multi-word phrases count as one item.
- Keep duplicates only if they appear with meaningfully different wording.
- Do not extract generic discourse words such as "step", "approach",
  "question", "answer" unless tied to a concrete option or shape.
- Respond strictly as one JSON object and nothing else.

JSON format:
{
  "target_related": [...],
  "distractor_related": [...],
  "other": [...],
  "counts": {
    "target_related": <int>,
    "distractor_related": <int>,
    "other": <int>
  }
}
""".strip()

ATTR_KEYS = [
    "target_related",
    "distractor_related",
    "other",
]

CODE_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", flags=re.I)
CODE_FENCE_CLOSE_RE = re.compile(r"\s*```$", flags=re.I)

_CLIENT: Optional[Any] = None


def get_client(api_key: str) -> Any:
    global _CLIENT
    if OpenAI is None:
        raise RuntimeError("openai package is not installed in this Python environment")
    if _CLIENT is None:
        kwargs = {"api_key": api_key, "base_url": DEEPSEEK_BASE_URL}
        if httpx is not None:
            kwargs["http_client"] = httpx.Client(
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            )
        _CLIENT = OpenAI(**kwargs)
    return _CLIENT


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = CODE_FENCE_OPEN_RE.sub("", text)
    text = CODE_FENCE_CLOSE_RE.sub("", text)
    return text.strip()


_BAD_BACKSLASH_RE = re.compile(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})')


def _repair_bad_escapes(text: str) -> str:
    """Double-escape any backslash not followed by a valid JSON escape token.
    Recovers most LaTeX-in-JSON cases (e.g., '\\parallel', '\\frac') that the
    model emits with a single backslash instead of '\\\\'.
    """
    return _BAD_BACKSLASH_RE.sub(r'\\\\', text)


def extract_first_json(text: str) -> Dict[str, Any]:
    text = strip_code_fences(text)
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object start", text, 0)
    depth = 0
    end = -1
    for idx, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        raise json.JSONDecodeError("Unbalanced braces", text, start)
    chunk = text[start:end]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return json.loads(_repair_bad_escapes(chunk))


def empty_result() -> Dict[str, Any]:
    return {**{k: [] for k in ATTR_KEYS}, "counts": {k: 0 for k in ATTR_KEYS}}


def build_user_message(obj: Dict[str, Any]) -> str:
    tgt_list = obj.get("target_labels") or []
    dst_list = obj.get("distractor_labels") or []
    tgt = ", ".join(str(x) for x in tgt_list) if tgt_list else "(unknown)"
    dst = ", ".join(str(x) for x in dst_list) if dst_list else "(unknown)"
    question = (
        obj.get("target_question")
        or obj.get("question_for_eval")
        or obj.get("question")
        or "(unknown)"
    )
    sentence = obj.get("answer", "")
    return (
        f"TARGET shapes: {tgt}\n"
        f"DISTRACTOR shapes: {dst}\n"
        f"QUESTION:\n{question}\n\n"
        f"Raw text:\n{sentence}"
    )


def generate_chat(client: OpenAI, user_content: str) -> str:
    last_err: Optional[Exception] = None
    for backoff in [0.0] + RETRY_BACKOFFS:
        if backoff > 0:
            time.sleep(backoff)
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=1e-6,
                max_tokens=2048,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
        except Exception as e:
            last_err = e
            break
    raise last_err if last_err else RuntimeError("Unknown API error")


def parse_sentence(client: OpenAI, obj: Dict[str, Any]) -> Dict[str, Any]:
    raw_reply = ""
    try:
        raw_reply = generate_chat(client, build_user_message(obj))
        parsed = extract_first_json(raw_reply)
        out = empty_result()
        for key in ATTR_KEYS:
            out[key] = list(parsed.get(key, []) or [])
        counts = parsed.get("counts") or {}
        for key in ATTR_KEYS:
            try:
                out["counts"][key] = int(counts.get(key, len(out[key])))
            except (TypeError, ValueError):
                out["counts"][key] = len(out[key])
        return out
    except Exception as e:
        out = empty_result()
        out["error"] = {
            "type": type(e).__name__,
            "msg": str(e),
            "raw_reply": raw_reply[:500],
        }
        return out


def _row_key(obj: Dict[str, Any]) -> Tuple[Any, Any]:
    return (obj.get("line_no"), obj.get("sentence_index"))


def _read_done_keys(output_file: Path) -> set:
    """Return set of (line_no, sentence_index) already in output_file."""
    done = set()
    if not output_file.exists() or output_file.stat().st_size == 0:
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(_row_key(json.loads(line)))
            except Exception:
                continue
    return done


def _annotate_obj(
    obj: Dict[str, Any],
    res: Dict[str, Any],
    idx: int,
    input_file: Path,
) -> None:
    for key in ATTR_KEYS:
        obj[key] = res.get(key, [])
    obj["counts"] = res.get("counts", {})
    if "error" in res:
        err = res["error"]
        obj["error_type"] = err.get("type")
        obj["error_msg"] = err.get("msg")
        obj["error_raw_reply"] = err.get("raw_reply")
        obj["error_line_index"] = idx
        obj["error_input_file"] = str(input_file)
        obj["error_model_id"] = MODEL_ID
        obj["error_prompt_tag"] = "mathverse_attribute_v1"
    else:
        for key in [
            "error_type", "error_msg", "error_raw_reply", "error_line_index",
            "error_input_file", "error_model_id", "error_prompt_tag",
        ]:
            obj.pop(key, None)


def convert_file(
    input_file: Path,
    output_file: Path,
    api_key: str,
    workers: int,
    resume: bool = True,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[int, Dict[str, Any]]] = []
    with input_file.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin, start=1):
            if line.strip():
                items.append((idx, json.loads(line)))

    if not items:
        if not output_file.exists():
            output_file.write_text("", encoding="utf-8")
        return

    done_keys = _read_done_keys(output_file) if resume else set()
    if done_keys:
        remaining = [(idx, obj) for idx, obj in items if _row_key(obj) not in done_keys]
        skipped = len(items) - len(remaining)
    else:
        remaining = items
        skipped = 0

    if not remaining:
        print(f"[SKIP] {output_file} fully done ({len(items)} rows)")
        return

    client = get_client(api_key)

    def run_one(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        idx, obj = item
        return idx, obj, parse_sentence(client, obj)

    write_mode = "a" if (resume and done_keys) else "w"
    n_err = 0
    with output_file.open(write_mode, encoding="utf-8", buffering=1) as fout, \
         ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, item) for item in remaining]
        for n_done, fut in enumerate(as_completed(futures), start=1):
            idx, obj, res = fut.result()
            _annotate_obj(obj, res, idx, input_file)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
            n_err += int("error" in res)
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(remaining)} done (errors={n_err})")

    print(f"[OK] {output_file} rows={len(items)} new={len(remaining)} resumed={skipped} errors={n_err}")


def make_error_subset(out_file: Path, err_file: Path) -> int:
    n_err = 0
    err_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("r", encoding="utf-8") as fin, err_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("error_msg"):
                for key in ATTR_KEYS + [
                    "counts", "error_type", "error_msg", "error_raw_reply",
                    "error_line_index", "error_input_file", "error_model_id", "error_prompt_tag",
                ]:
                    obj.pop(key, None)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_err += 1
    return n_err


def loop_convert(
    input_file: Path,
    out_prefix: Path,
    api_key: str,
    workers: int,
    max_rounds: int,
    resume: bool = True,
) -> None:
    cur_input = input_file
    for round_idx in range(max_rounds):
        out_file = Path(f"{out_prefix}.r{round_idx}.out.jsonl")
        err_file = Path(f"{out_prefix}.r{round_idx}.errors.jsonl")
        convert_file(cur_input, out_file, api_key, workers, resume=resume)
        n_err = make_error_subset(out_file, err_file)
        if n_err == 0:
            return
        print(f"[RETRY] round={round_idx} errors={n_err}")
        cur_input = err_file


def out_prefix_for(input_file: Path, input_root: Optional[Path], out_root: Path) -> Path:
    """Mirror input_file's path under out_root (relative to input_root if given)."""
    if input_root is None:
        rel = Path(input_file.name)
    else:
        rel = input_file.resolve().relative_to(input_root.resolve())
    return (out_root / rel).with_suffix("")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, help="directory of per-sentence JSONL files (or use --files)")
    p.add_argument("--files", nargs="+", type=Path)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--out_prefix", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_rounds", type=int, default=2)
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="resume: skip rows already in output file (matched by line_no+sentence_index); append the rest. Without this flag the output file is overwritten.",
    )
    p.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY"))
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("[ERR] set DEEPSEEK_API_KEY or pass --api_key")

    resume = args.skip_existing
    if not args.files and args.input is None:
        raise SystemExit("[ERR] provide --input (directory or file) or --files")
    if args.files:
        input_root = args.input
        files = args.files
        for idx, fp in enumerate(files, start=1):
            prefix = out_prefix_for(fp, input_root, args.out_dir)
            print(f"[{idx}/{len(files)}] {fp}")
            loop_convert(fp, prefix, args.api_key, args.workers, args.max_rounds, resume=resume)
    elif args.input.is_dir():
        files = sorted(args.input.glob("*.jsonl"))
        for idx, fp in enumerate(files, start=1):
            prefix = out_prefix_for(fp, args.input, args.out_dir)
            print(f"[{idx}/{len(files)}] {fp}")
            loop_convert(fp, prefix, args.api_key, args.workers, args.max_rounds, resume=resume)
    else:
        if not args.out_prefix:
            raise SystemExit("[ERR] file mode requires --out_prefix")
        loop_convert(args.input, args.out_prefix, args.api_key, args.workers, args.max_rounds, resume=resume)


if __name__ == "__main__":
    main()
