#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Waterbirds results → Excel (accuracy by background / bias group, reasoning length).

  input  : run_perception.py --dataset waterbirds outputs under <root>/<model>/
  usage  : python score_waterbirds.py --root <dir> --out <xlsx>
"""

import re
import json
import argparse
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

import pandas as pd

# model family (directory / filename key) → display name
DISPLAY_NAME = {
    "qwen3-thinking": "Qwen3-thinking",
    "glm":            "GLM",
    "intern":         "Intern",
    "onevision":      "OneVision",
}
MODELS_ORDER = ["Qwen3-thinking", "GLM", "Intern", "OneVision"]
FAMILY_DIRS = list(DISPLAY_NAME)

TOKENIZER_PATH = {
    "qwen3-thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "glm":            "zai-org/GLM-4.1V-9B-Thinking",
    "intern":         "internlm/Intern-S1-mini",
    "onevision":      "Fancy-MLLM/R1-Onevision-7B-RL",
}
DATASET_NAME = "Waterbirds"

# valid labels
ALLOWED = {"dog","bird","vehicle","reptile","carnivore","insect","instrument","primate","fish",
           "landbird","waterbird"}

# ===== Tokenizer / Processor =====
try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

@lru_cache(maxsize=None)
def load_processor(model_path: str):
    if AutoProcessor is None:
        raise RuntimeError("transformers is not installed")
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def count_tokens(text: str, processor) -> int:
    if not text:
        return 0
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    if hasattr(processor, "encode"):
        return len(processor.encode(text, add_special_tokens=False))
    raise RuntimeError("no tokenizer/encode in processor")

def safe_count_tokens(text: str, family_key: str) -> int:
    try:
        model_path = TOKENIZER_PATH[family_key]
        proc = load_processor(model_path)
        return count_tokens(text or "", proc)
    except Exception:
        return len(text.split()) if isinstance(text, str) else 0

# ====== answer extraction patterns ======
ANS_OPEN_RE  = re.compile(r'(?is)<\s*answer\s*>')
ANS_CLOSE_RE = re.compile(r'(?is)</\s*answer\s*>')
BOX_OPEN_RE  = re.compile(r'(?s)\<\|begin_of_box\|\>')
BOX_CLOSE_RE = re.compile(r'(?s)\<\|end_of_box\|\>')
PART_RE      = re.compile(r"_part_(\d+)\.jsonl$", re.IGNORECASE)

# R1-OneVision often omits the tags: "Answer: Landbird", "classified as a landbird", or a bare landbird/waterbird mention
ANSWER_AFTER_RE   = re.compile(r'(?im)(?:^|\n)\s*(?:final\s+)?answer\s*[:：]\s*([^\n\r<]+)')
CLASSIFIED_AS_RE  = re.compile(r'(?i)classified\s+as\s+(?:a\s+)?(landbird|waterbird)')
LANDWATER_ANY_RE  = re.compile(r'(?i)\b(land\s*bird|water\s*bird|landbird|waterbird)\b')

# ====== helpers ======
def acc_of(c, t): return 0.0 if t == 0 else 100.0 * (c / t)

def _normalize_first_token(s: str) -> str:
    if s is None: return ""
    raw = s.strip().lower()
    for sep in ["/", ",", "|"]:
        if sep in raw:
            raw = raw.split(sep)[0].strip()
            break
    return raw

def _canon_landwater(s: str):
    if not isinstance(s, str): return None
    t = re.sub(r'[^a-z]', '', s.strip().lower())
    if 'landbird' in t: return 'landbird'
    if 'waterbird' in t: return 'waterbird'
    return None

def pick_label_from_free_text(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    last = None
    for m in ANSWER_AFTER_RE.finditer(text):
        cand = _canon_landwater(m.group(1))
        if cand: last = cand
    if last: return last
    m = CLASSIFIED_AS_RE.search(text)
    if m:
        return _canon_landwater(m.group(1))
    matches = list(LANDWATER_ANY_RE.finditer(text))
    if matches:
        return _canon_landwater(matches[-1].group(1))
    return None

def canonical_label(s: str):
    if not isinstance(s, str): return None
    t = s.strip().lower().replace(" ", "").replace("-", "")
    if t in {"landbird","waterbird"}: return t
    return t if t in ALLOWED else t

def extract_from_answer_tags(answer_text: str):
    if not isinstance(answer_text, str) or not answer_text:
        return "null", None, None, "no_answer_field"
    opens = list(ANS_OPEN_RE.finditer(answer_text))
    closes = list(ANS_CLOSE_RE.finditer(answer_text))
    if not opens and not closes:
        ft = _normalize_first_token(answer_text)
        return (ft if ft else "null"), answer_text, "no_tag_first_token", None

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
    norm0 = canonical_label(_normalize_first_token(cand_texts[0][1]))
    if norm0 in ALLOWED or norm0 in {"landbird","waterbird"}:
        return norm0, cand_texts[0][1], "last", None
    if len(cand_texts) >= 2:
        norm1 = canonical_label(_normalize_first_token(cand_texts[1][1]))
        if norm1 in ALLOWED or norm1 in {"landbird","waterbird"}:
            return norm1, cand_texts[1][1], "prev", None
        return norm0, cand_texts[0][1], "last_unmatched_keep", "unmatched_last_prev"
    return norm0, cand_texts[0][1], "last_unmatched_keep", "unmatched_last"

def extract_from_box(answer_text: str):
    if not isinstance(answer_text, str) or not answer_text:
        return "null", None, None, "no_box_field"
    opens = list(BOX_OPEN_RE.finditer(answer_text))
    closes = list(BOX_CLOSE_RE.finditer(answer_text))
    if not opens:
        return extract_from_answer_tags(answer_text)
    start = opens[-1].end()
    endpos = None
    for cm in closes:
        if cm.start() >= start:
            endpos = cm.start()
            break
    if endpos is None:
        endpos = len(answer_text)
    content = answer_text[start:endpos].strip()
    norm = canonical_label(_normalize_first_token(content))
    return (norm if norm else "null"), content, "last_box", None

def detect_family_key_from_path(p: str) -> str:
    low = p.lower()
    if "qwen3-thinking" in low: return "qwen3-thinking"
    if "glm" in low: return "glm"
    if "intern" in low: return "intern"
    if "onevision" in low: return "onevision"
    return "unknown"

def norm_background(x, path_hint=None):
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"land","water"}: return t
    if path_hint:
        stem = Path(path_hint).stem.lower()
        if "land" in stem and "water" not in stem: return "land"
        if "water" in stem and "land" not in stem: return "water"
        for part in map(str.lower, Path(path_hint).parts):
            if part in {"land","water"}: return part
            if "-land" in part and "water" not in part: return "land"
            if "-water" in part and "land" not in part: return "water"
    return None

def is_conflicting(label, background):
    return (label == "landbird" and background == "water") or \
           (label == "waterbird" and background == "land")

def is_aligned(label, background):
    return (label == "landbird" and background == "land") or \
           (label == "waterbird" and background == "water")

def list_models(root: Path):
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def list_parts(model_dir: Path):
    files = []
    for p in model_dir.glob("*.jsonl"):
        m = PART_RE.search(p.name)
        if m:
            try:
                idx = int(m.group(1))
            except Exception:
                idx = 999999
            files.append((idx, p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files] or sorted(model_dir.glob("*.jsonl"))

# ====== main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="waterbirds results root: <root>/<model>/*_part_NN.jsonl")
    ap.add_argument("--out", required=True, help="output .xlsx path")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # accumulators: correct / total / token sum
    per_model = defaultdict(lambda: {"correct": 0, "total": 0, "tok_sum": 0})
    per_lb    = defaultdict(lambda: {"correct": 0, "total": 0, "tok_sum": 0})  # (model, label, background)
    per_ac    = defaultdict(lambda: {"correct": 0, "total": 0, "tok_sum": 0})  # (model, "Aligned"/"Conflicting")

    models = list_models(root)
    if not models:
        print(f"[WARN] No model directories under: {root}")

    for mdir in models:
        fam_key = detect_family_key_from_path(mdir.name)
        model_disp = DISPLAY_NAME.get(fam_key, mdir.name)

        parts = list_parts(mdir)
        if not parts:
            print(f"[WARN] No jsonl files for model: {mdir.name}")
            continue

        line_no_global = 0
        for jf in parts:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line_no_global += 1
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    label_raw = obj.get("label")
                    label = canonical_label(label_raw)
                    bg = norm_background(obj.get("background") or obj.get("bg") or obj.get("bg_type") or obj.get("background_type"),
                                         path_hint=str(jf))
                    ans_text = obj.get("answer")

                    # final answer: GLM uses <|begin_of_box|>, the others <answer> tags
                    if fam_key == "glm":
                        fa, raw, used_idx, err = extract_from_box(ans_text)
                        source = "box"
                    else:
                        fa, raw, used_idx, err = extract_from_answer_tags(ans_text)
                        source = "answer_tag"

                    # an explicit final_answer field takes precedence
                    fa_override = obj.get("final_answer")
                    if isinstance(fa_override, str) and fa_override.strip():
                        fa = canonical_label(fa_override)

                    # R1-OneVision fallback: free-text answer patterns
                    if fam_key == "onevision":
                        ov = pick_label_from_free_text(ans_text or "") or pick_label_from_free_text(raw or "")
                        if ov:
                            fa = ov

                    fa = canonical_label(fa)
                    correct = int(fa is not None and label is not None and fa == label)

                    # reasoning length under the model's own tokenizer
                    token_len = safe_count_tokens(ans_text or "", fam_key)

                    # accumulate
                    per_model[model_disp]["total"]   += 1
                    per_model[model_disp]["correct"] += correct
                    per_model[model_disp]["tok_sum"] += token_len

                    if label in {"landbird","waterbird"} and bg in {"land","water"}:
                        per_lb[(model_disp, label, bg)]["total"]   += 1
                        per_lb[(model_disp, label, bg)]["correct"] += correct
                        per_lb[(model_disp, label, bg)]["tok_sum"] += token_len

                        if is_aligned(label, bg):
                            per_ac[(model_disp, "Aligned")]["total"]   += 1
                            per_ac[(model_disp, "Aligned")]["correct"] += correct
                            per_ac[(model_disp, "Aligned")]["tok_sum"] += token_len
                        elif is_conflicting(label, bg):
                            per_ac[(model_disp, "Conflicting")]["total"]   += 1
                            per_ac[(model_disp, "Conflicting")]["correct"] += correct
                            per_ac[(model_disp, "Conflicting")]["tok_sum"] += token_len

                    rows.append({
                        "dataset": DATASET_NAME,
                        "model": model_disp,
                        "family": fam_key,
                        "path": str(jf),
                        "line_no": line_no_global,
                        "label": label,
                        "background": bg,
                        "final_answer": fa,
                        "final_answer_raw": raw,
                        "source": source,
                        "used_idx": used_idx,
                        "parse_error": err,
                        "correct": correct,
                        "token_len": token_len,
                    })

        # console summary
        def avg_tl(d): return 0.0 if d["total"] == 0 else (d["tok_sum"] / d["total"])
        pm = per_model[model_disp]
        print(f"\n=== Model: {model_disp} ===")
        print(f"OVERALL     -> ACC: {acc_of(pm['correct'], pm['total']):.2f}% "
              f"({pm['correct']}/{pm['total']}), avg TL: {avg_tl(pm):.2f}")

        print("=== Per (label, background) ===")
        order = [
            ("landbird", "land"),
            ("landbird", "water"),
            ("waterbird", "land"),
            ("waterbird", "water"),
        ]
        for k in order:
            key = (model_disp, k[0], k[1])
            c = per_lb[key]["correct"]; t = per_lb[key]["total"]
            tok_mean = 0.0 if t == 0 else per_lb[key]["tok_sum"] / t
            print(f"label: {k[0]:9s} - background: {k[1]:5s} -> "
                  f"ACC: {acc_of(c,t):.2f}% ({c}/{t}), avg TL: {tok_mean:.2f}")

        print("\n=== Align / Conflicting ===")
        for grp in ["Aligned","Conflicting"]:
            c = per_ac[(model_disp, grp)]["correct"]; t = per_ac[(model_disp, grp)]["total"]
            tok_mean = 0.0 if t == 0 else per_ac[(model_disp, grp)]["tok_sum"] / t
            print(f"{grp.upper():12s}-> ACC: {acc_of(c,t):.2f}% ({c}/{t}), avg TL: {tok_mean:.2f}")

    # ===== write Excel =====
    df_raw = pd.DataFrame(rows)

    # overall
    overall_rows = []
    for model_name, cnt in per_model.items():
        t = cnt["total"]; c = cnt["correct"]; tok_mean = 0.0 if t == 0 else cnt["tok_sum"]/t
        overall_rows.append({"model": model_name,
                             "acc": round((c/t) if t else 0.0, 6),
                             "avg_token_length": round(tok_mean, 3),
                             "correct": c, "total": t})
    df_overall = pd.DataFrame(overall_rows)
    if not df_overall.empty:
        order_map = {m:i for i,m in enumerate(MODELS_ORDER)}
        df_overall["order"] = df_overall["model"].map(order_map).fillna(9999)
        df_overall = df_overall.sort_values(["order","model"]).drop(columns=["order"])

    # per (label, background)
    plb_rows = []
    for (model_name, lbl, bg), cnt in per_lb.items():
        t = cnt["total"]; c = cnt["correct"]; tok_mean = 0.0 if t == 0 else cnt["tok_sum"]/t
        plb_rows.append({"model": model_name, "label": lbl, "background": bg,
                         "acc": round((c/t) if t else 0.0, 6),
                         "avg_token_length": round(tok_mean, 3),
                         "correct": c, "total": t})
    df_plb = pd.DataFrame(plb_rows)
    if not df_plb.empty:
        lbl_order = {("landbird","land"):0,("landbird","water"):1,("waterbird","land"):2,("waterbird","water"):3}
        df_plb["pair_order"] = df_plb.apply(lambda r: lbl_order.get((r["label"], r["background"]), 9999), axis=1)
        df_plb = df_plb.sort_values(["model","pair_order"]).drop(columns=["pair_order"])

    # aligned / conflicting
    ac_rows = []
    for (model_name, grp), cnt in per_ac.items():
        t = cnt["total"]; c = cnt["correct"]; tok_mean = 0.0 if t == 0 else cnt["tok_sum"]/t
        ac_rows.append({"model": model_name, "group": grp,
                        "acc": round((c/t) if t else 0.0, 6),
                        "avg_token_length": round(tok_mean, 3),
                        "correct": c, "total": t})
    df_ac = pd.DataFrame(ac_rows)
    if not df_ac.empty:
        df_ac["group_order"] = df_ac["group"].map({"Aligned": 0, "Conflicting": 1}).fillna(2)
        df_ac = df_ac.sort_values(["model","group_order"]).drop(columns=["group_order"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_raw.to_excel(writer, index=False, sheet_name="predictions_raw")
        df_overall.to_excel(writer, index=False, sheet_name="overall")
        df_plb.to_excel(writer, index=False, sheet_name="per_label_background")
        df_ac.to_excel(writer, index=False, sheet_name="align_conflict")

    print(f"\n[OK] Wrote Excel: {out_path}")

if __name__ == "__main__":
    main()
