#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Idis-perception results → Excel (accuracy, reasoning length).

  input  : run_perception.py outputs under <root>  (model family inferred from the path)
  answer : <answer>X</answer>;  <|begin_of_box|> for GLM
  usage  : python score_perception.py --root <dir> --out <xlsx> [--dataset-name ..] [--cases Conflicting]
"""

import os
import re
import json
import argparse
from pathlib import Path
from functools import lru_cache

import pandas as pd

# ===== Tokenizer / Processor =====
try:
    from transformers import AutoProcessor
except Exception:  # keep the script importable without transformers (whitespace-token fallback)
    AutoProcessor = None

# model family (directory / filename key) → display name
DISPLAY_NAME = {
    "qwen3-thinking": "Qwen3-thinking",
    "glm":            "GLM",
    "intern":         "Intern",
    "onevision":      "OneVision",
}
MODELS_ORDER = ["Qwen3-thinking", "GLM", "Intern", "OneVision"]
FAMILY_DIRS = list(DISPLAY_NAME)

# model family → tokenizer used for reasoning-length counts
TOKENIZER_PATH = {
    "qwen3-thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "glm":            "zai-org/GLM-4.1V-9B-Thinking",
    "intern":         "internlm/Intern-S1-mini",
    "onevision":      "Fancy-MLLM/R1-Onevision-7B-RL",
}
CASES_ORDER = ["Aligned", "Conflicting", "Irrelevant"]   # overridden by --cases
DATASET_NAME = "Idis-perception"                          # overridden by --dataset-name

# valid labels
ALLOWED = {"dog","bird","vehicle","reptile","carnivore","insect","instrument","primate","fish"}

# ====== answer extraction patterns ======
ANS_OPEN_RE  = re.compile(r'(?is)<\s*answer\s*>')
ANS_CLOSE_RE = re.compile(r'(?is)</\s*answer\s*>')
BOX_OPEN_RE  = re.compile(r'(?s)\<\|begin_of_box\|\>')
BOX_CLOSE_RE = re.compile(r'(?s)\<\|end_of_box\|\>')

# ====== tokenizer loading / token counting ======
@lru_cache(maxsize=None)
def load_processor(model_path: str):
    if AutoProcessor is None:
        raise RuntimeError("transformers is not installed (pip install transformers)")
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def count_tokens(text: str, processor) -> int:
    if not text:
        return 0
    # most processors expose a tokenizer
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    # some processors implement encode() directly
    if hasattr(processor, "encode"):
        return len(processor.encode(text, add_special_tokens=False))
    raise RuntimeError("processor has neither a tokenizer nor encode()")

def safe_count_tokens(text: str, family_key: str) -> int:
    """Token length under the model family's tokenizer (lazy-loaded); falls back to whitespace tokens."""
    try:
        model_path = TOKENIZER_PATH.get(family_key)
        if not model_path:
            raise KeyError(f"unknown model family: {family_key}")
        proc = load_processor(model_path)
        return count_tokens(text, proc)
    except Exception:
        # fallback: whitespace token count
        return len(text.split()) if isinstance(text, str) else 0

# ====== helpers ======
def _normalize_first_token(s: str) -> str:
    if s is None:
        return ""
    raw = s.strip().lower()
    for sep in ["/", ",", "|"]:
        if sep in raw:
            raw = raw.split(sep)[0].strip()
            break
    return raw

def extract_from_answer_tags(answer_text: str):
    if not isinstance(answer_text, str) or not answer_text:
        return "null", None, None, "no_answer_field"
    opens = list(ANS_OPEN_RE.finditer(answer_text))
    closes = list(ANS_CLOSE_RE.finditer(answer_text))
    if not opens and not closes:
        return "null", None, None, "no_answer_tag"

    def slice_from_open(open_m):
        start = open_m.end()
        endpos = None
        for cm in closes:
            if cm.start() >= start:
                endpos = cm.start()
                break
        if endpos is None:
            endpos = len(answer_text)
        return answer_text[start:endpos].strip()

    candidates = []
    if opens:
        candidates.append(("last", opens[-1]))
        if len(opens) >= 2:
            candidates.append(("prev", opens[-2]))
    elif closes:
        before = answer_text[:closes[-1].start()]
        content = before.strip()
        norm = _normalize_first_token(content)
        return (norm if norm else "null"), content, "no_open_before_last_close", None

    cand_texts = [(tag, slice_from_open(m)) for tag, m in candidates]

    norm0 = _normalize_first_token(cand_texts[0][1])
    if norm0 in ALLOWED:
        return norm0, cand_texts[0][1], "last", None
    if len(cand_texts) >= 2:
        norm1 = _normalize_first_token(cand_texts[1][1])
        if norm1 in ALLOWED:
            return norm1, cand_texts[1][1], "prev", None
        return norm0, cand_texts[0][1], "last_unmatched_keep", "unmatched_last_prev"
    return norm0, cand_texts[0][1], "last_unmatched_keep", "unmatched_last"

def extract_from_box(answer_text: str):
    if not isinstance(answer_text, str) or not answer_text:
        return "null", None, None, "no_box_field"
    opens = list(BOX_OPEN_RE.finditer(answer_text))
    closes = list(BOX_CLOSE_RE.finditer(answer_text))
    if not opens:
        return "null", None, None, "no_box_tag"
    last_open = opens[-1]
    start = last_open.end()
    endpos = None
    for cm in closes:
        if cm.start() >= start:
            endpos = cm.start()
            break
    if endpos is None:
        endpos = len(answer_text)
    content = answer_text[start:endpos].strip()
    norm = _normalize_first_token(content)
    return (norm if norm else "null"), content, "last_box", None

def parse_case_n_from_filename(path: str):
    """Parse n_objects and case from a filename such as ...-3-aligned.jsonl."""
    stem = Path(path).stem.lower()
    parts = stem.split("-")
    n_obj, case = None, None
    if len(parts) >= 2:
        last = parts[-1]
        prev = parts[-2]
        if last in ("aligned","conflicting","irrelevant"):
            case = last.capitalize()
        if prev.isdigit():
            n_obj = int(prev)
    return n_obj, case

def infer_case_from_any(p: str):
    _, case_from_name = parse_case_n_from_filename(p)
    if case_from_name:
        return case_from_name
    low = p.lower()
    for c in ("aligned","conflicting","irrelevant"):
        if f"/{c}" in low or low.endswith(c+".jsonl"):
            return c.capitalize()
    return None

def infer_n_objects_any(obj: dict, path: str):
    if isinstance(obj.get("n_objects"), int):
        return obj["n_objects"]
    n_from_name, _ = parse_case_n_from_filename(path)
    if n_from_name is not None:
        return n_from_name
    for part in Path(path).parts:
        if part.isdigit():
            return int(part)
    return None

def detect_family_key_from_path(p: str) -> str:
    low = p.lower()
    if "qwen3-thinking" in low: return "qwen3-thinking"
    if "glm" in low: return "glm"
    if "intern" in low: return "intern"
    if "onevision" in low: return "onevision"
    return "unknown"

def process_jsonl(file_path: Path):
    fam = detect_family_key_from_path(str(file_path))
    rows = []
    if not file_path.exists():
        return rows
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            ans_text = obj.get("answer")
            label = obj.get("label")

            # final answer: GLM uses <|begin_of_box|>, the others <answer> tags
            if fam == "glm":
                fa, raw, used_idx, err = extract_from_box(ans_text)
                source = "box"
            else:
                fa, raw, used_idx, err = extract_from_answer_tags(ans_text)
                source = "answer_tag"

            # case-insensitive comparison with the label
            label_norm = (label.lower().strip() if isinstance(label, str) else None)
            fa_norm = (fa.lower().strip() if isinstance(fa, str) else fa)
            correct = int(fa_norm != "null" and label_norm is not None and fa_norm == label_norm)

            # reasoning length under the model's own tokenizer
            token_len = safe_count_tokens(ans_text or "", fam)

            n_objs = infer_n_objects_any(obj, str(file_path))
            case_val = infer_case_from_any(str(file_path)) or "Unknown"

            rows.append({
                "dataset": DATASET_NAME,
                "model": DISPLAY_NAME.get(fam, fam),
                "family": fam,
                "case": case_val,
                "n_objects": n_objs,
                "acc_sample": correct,
                "token_len": token_len,
                "path": str(file_path),
                "line_no": i,
                "label": label_norm,
                "final_answer": fa_norm,
                "final_answer_raw": raw,
                "source": source,
                "used_idx": used_idx,
                "parse_error": err,
            })
    return rows

def walk_and_collect(root: Path):
    """Recursively collect every *.jsonl under root whose path names a known model family."""
    all_rows = []
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        print(f"[WARN] no jsonl files under {root}")
    for jf in files:
        if detect_family_key_from_path(str(jf)) != "unknown":
            all_rows += process_jsonl(jf)
    return all_rows

# ====== per-model sheet ======
def build_model_sheet(agg: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for case in CASES_ORDER:
        n_values = (agg.loc[(agg["model"]==model_name) & (agg["case"]==case), "n_objects"]
                      .dropna().drop_duplicates().sort_values().tolist())
        if not n_values:
            continue
        for n in n_values:
            sub = agg[(agg["model"]==model_name) & (agg["case"]==case) & (agg["n_objects"]==n)]
            r = sub.iloc[0]
            rows.append({
                "Case": case,
                "# of distractor": int(n) if pd.notna(n) else None,
                "acc": round(float(r["acc"]), 6),
                "avg token length": round(float(r["avg_token_length"]), 3)
            })
    return pd.DataFrame(rows, columns=["Case","# of distractor","acc","avg token length"])

# ====== main ======
def main():
    global CASES_ORDER, DATASET_NAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory containing per-model result *.jsonl (searched recursively)")
    ap.add_argument("--out", required=True, help="output .xlsx path")
    ap.add_argument("--dataset-name", default=DATASET_NAME, help="value written to the 'dataset' column")
    ap.add_argument("--cases", default=",".join(CASES_ORDER),
                    help="comma-separated case order for the per-model sheets")
    args = ap.parse_args()
    DATASET_NAME = args.dataset_name
    CASES_ORDER = [c.strip() for c in args.cases.split(",") if c.strip()]

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = walk_and_collect(root)
    if not rows:
        print(f"[WARN] No JSONL rows found under: {root}")

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # raw per-sample rows
        df.to_excel(writer, index=False, sheet_name="predictions_raw")

        # aggregate by (dataset, model, case, n_objects)
        agg = (df.groupby(["dataset","model","case","n_objects"], as_index=False)
                 .agg(acc=("acc_sample","mean"),
                      avg_token_length=("token_len","mean"),
                      num_samples=("acc_sample","size")))

        # one sheet per model
        for model_name in MODELS_ORDER:
            sub = agg[agg["model"] == model_name]
            if sub.empty:
                pd.DataFrame(columns=["Case","# of distractor","acc","avg token length"])\
                    .to_excel(writer, index=False, sheet_name=model_name)
            else:
                build_model_sheet(agg, model_name).to_excel(writer, index=False, sheet_name=model_name)

    print(f"[OK] Wrote Excel: {out_path}")

if __name__ == "__main__":
    main()
