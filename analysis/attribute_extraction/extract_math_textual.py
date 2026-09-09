#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek attribute extraction for Idis-math textual-distractor traces (Table 7 prompt).

  input  : JSONL rows {task, sample_index, problem_version, label, question,
                       distractor_statements, sentences}
  output : one row per sentence, target_related / distractor_related / other, counts
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
WORK_ROOT = Path(__file__).parent

SYSTEM_PROMPT = """
You are analyzing a model's mathematical reasoning trace for the MathVerse
textual-distractor setting, where the ORIGINAL question (the TARGET problem)
was augmented by inserting one or more DISTRACTOR statements: plausible-
sounding but irrelevant or conflicting conditions that are NOT needed to
answer the question. The reasoning trace was produced by a model that saw the
augmented question.

The user message provides three pieces of context for this sample:
  QUESTION:              <the original problem statement, without distractors>
  DISTRACTOR statements: <the inserted sentences, numbered D1, D2, ...>
  Raw text:              a single sentence from the model's reasoning trace.

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
- target_related: phrases that refer to (a) any entity, property, measurement,
  relation, value, or option mentioned in the QUESTION, OR (b) a part/property/
  operation that the surrounding sentence clearly attaches to the TARGET
  problem (e.g. a vertex, side, angle, area, congruence claim, or theorem
  usage about the TARGET).
- distractor_related: phrases that refer to an entity, value, measurement, or
  relation introduced by a DISTRACTOR statement — the Raw text names it, uses
  its value, reasons about it, or explicitly dismisses it. If the same entity
  also appears in the QUESTION, it is target_related, not distractor_related.
- other: concrete attribute phrases that cannot be confidently assigned to
  either the TARGET or a DISTRACTOR — generic math terminology, unrelated
  entities, or otherwise ambiguous mentions.

Rules:
- Use only words/phrases that literally appear in the Raw text. Do not infer,
  translate, or paraphrase. The QUESTION and DISTRACTOR statements are for
  grounding only — do not extract from them.
- A phrase counts as target_related if it names or clearly refers to anything
  introduced by the QUESTION. Anaphora ("it", "the segment") counts only when
  the referent is unambiguous from the sentence.
- A phrase counts as distractor_related only when the Raw text makes the link
  to a DISTRACTOR statement explicit (e.g. names its entity, uses its value,
  or compares it against the TARGET).
- If the link is ambiguous, put the phrase in "other".
- Multi-word phrases count as one item.
- Keep duplicates only if they appear with meaningfully different wording.
- Do not extract generic discourse words such as "step", "approach",
  "question", "answer" unless tied to a concrete option or entity.
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

ATTR_KEYS = ["target_related", "distractor_related", "other"]

CODE_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", flags=re.I)
CODE_FENCE_CLOSE_RE = re.compile(r"\s*```$", flags=re.I)
_BAD_BACKSLASH_RE = re.compile(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})')

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


def _repair_bad_escapes(text: str) -> str:
    return _BAD_BACKSLASH_RE.sub(r"\\\\", text)


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


def build_user_message(item: Dict[str, Any]) -> str:
    stmts = "\n".join(f"D{k}. {s}" for k, s in
                      enumerate(item["distractor_statements"], start=1))
    return (
        f"QUESTION:\n{item['question']}\n\n"
        f"DISTRACTOR statements:\n{stmts}\n\n"
        f"Raw text:\n{item['answer']}"
    )


def generate_chat(client: Any, user_content: str) -> str:
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


def parse_sentence(client: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    raw_reply = ""
    try:
        raw_reply = generate_chat(client, build_user_message(item))
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


def expand_traces(trace_rows: List[Dict], min_len: int = 1) -> List[Dict]:
    """One work item per sentence, mirroring the visual per-sentence rows."""
    items = []
    for row in trace_rows:
        for sent_idx, sent in enumerate(row["sentences"], start=1):
            if len(sent.split()) < min_len:
                continue
            items.append({
                "task": row["task"],
                "sample_index": row["sample_index"],
                "problem_version": row["problem_version"],
                "label": row.get("label", ""),
                "question": row["question"],
                "distractor_statements": row["distractor_statements"],
                "sentence_index": sent_idx,
                "n_sentences": row["n_sentences"],
                "answer": sent,
            })
    return items


def row_key(obj: Dict[str, Any]) -> Tuple:
    return (obj.get("task"), str(obj.get("sample_index")),
            obj.get("problem_version"), int(obj.get("sentence_index", -1)))


def read_done_keys(output_file: Path) -> set:
    """Keys already answered successfully (error rows are retried on rerun)."""
    done = set()
    if not output_file.exists() or output_file.stat().st_size == 0:
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not obj.get("error_msg"):
                done.add(row_key(obj))
    return done


def annotate(obj: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    for key in ATTR_KEYS:
        out[key] = res.get(key, [])
    out["counts"] = res.get("counts", {})
    if "error" in res:
        err = res["error"]
        out["error_type"] = err.get("type")
        out["error_msg"] = err.get("msg")
        out["error_raw_reply"] = err.get("raw_reply")
        out["error_model_id"] = MODEL_ID
        out["error_prompt_tag"] = "mathverse_textual_attribute_v1"
    return out


def convert_file(input_file: Path, output_file: Path, api_key: str,
                 workers: int, limit_traces: Optional[int]) -> None:
    trace_rows = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trace_rows.append(json.loads(line))
    if limit_traces:
        trace_rows = trace_rows[:limit_traces]

    items = expand_traces(trace_rows)
    done = read_done_keys(output_file)
    remaining = [it for it in items if row_key(it) not in done]
    print(f"[{input_file.name}] traces={len(trace_rows)} sentences={len(items)} "
          f"done={len(done)} todo={len(remaining)} workers={workers}", flush=True)
    if not remaining:
        return

    client = get_client(api_key)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_err, k, t0 = 0, 0, time.time()
    batch_size = max(2000, workers * 50)
    write_mode = "a" if done else "w"
    with output_file.open(write_mode, encoding="utf-8", buffering=1) as fout, \
         ThreadPoolExecutor(max_workers=workers) as ex:
        for b0 in range(0, len(remaining), batch_size):
            batch = remaining[b0:b0 + batch_size]
            futures = [ex.submit(lambda it=it: (it, parse_sentence(client, it)))
                       for it in batch]
            for fut in as_completed(futures):
                item, res = fut.result()
                fout.write(json.dumps(annotate(item, res), ensure_ascii=False) + "\n")
                n_err += int("error" in res)
                k += 1
                if k % 500 == 0 or k == len(remaining):
                    rate = k / max(1e-9, time.time() - t0)
                    eta_h = (len(remaining) - k) / max(1e-9, rate) / 3600
                    print(f"  {k}/{len(remaining)} errors={n_err} "
                          f"({rate:.1f} sent/s, ETA {eta_h:.1f}h)", flush=True)

    print(f"[OK] {output_file} new={len(remaining)} errors={n_err} "
          f"(rerun to retry errors)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", type=Path, required=True,
                   help="trace JSONL files (schema: see module docstring)")
    p.add_argument("--out_dir", type=Path, default=WORK_ROOT / "deepseek_out")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-traces", type=int, default=None,
                   help="cap #traces per input file (smoke tests)")
    p.add_argument("--api_key",
                   default=os.environ.get("DEEPSEEK_API_KEY")
                  )
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("[ERR] set DEEPSEEK_API_KEY or pass --api_key")

    for fp in args.input:
        out_file = args.out_dir / fp.name.replace(".jsonl", ".out.jsonl")
        convert_file(fp, out_file, args.api_key, args.workers, args.limit_traces)


if __name__ == "__main__":
    main()
