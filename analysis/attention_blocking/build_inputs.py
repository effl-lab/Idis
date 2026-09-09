"""Inputs for the attention-blocking experiment.

  input  : DeepSeek sentence annotations + original reasoning traces (n=4, conflicting)
  output : parquet, one row per image
           pre_answer (trace cut after <answer>), dist_spans, tgt_spans (char spans)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────
MAIN_CLASSES = ["dog", "bird", "vehicle", "reptile", "carnivore",
                "insect", "instrument", "primate", "fish"]
LABEL_ALIASES = {"wheeled vehicle": "vehicle", "musical instrument": "instrument"}
SUPPORTED_MODELS = ["qwen3-thinking", "glm", "intern", "onevision"]

_CLASS_WORD_RES = [re.compile(rf"\b{c}s?\b", re.I) for c in MAIN_CLASSES]
PROMPT_RESTATE_THRESHOLD = 5          # ≥5 class words in one sentence → prompt restatement
ANS_OPEN_RE = re.compile(r"(?is)<\s*answer\s*>")
THINK_CLOSE_RE = re.compile(r"\n</think>")
BEGIN_BOX_RE = re.compile(r"<\|begin_of_box\|>")   # GLM answer marker



# ── label / sentence helpers ───────────────────────────────────────────────
def normalize_label(s: str) -> str:
    s = (s or "").strip().lower().replace("_", " ")
    return LABEL_ALIASES.get(s, s)


def is_prompt_restating(text: str) -> bool:
    if not text:
        return False
    return sum(1 for r in _CLASS_WORD_RES if r.search(text)) >= PROMPT_RESTATE_THRESHOLD


def classify_sentence(srec: dict, target_label: str) -> list[str]:
    """Return the sentence types a DeepSeek-annotated sentence qualifies for.

      distractor : counts[non-target class] > 0
      target_attr: counts[target class]     > 0
    A mixed sentence qualifies for both.  Empty / prompt-restating sentences → [].
    """
    text = srec.get("answer", "") or ""
    if not text.strip() or is_prompt_restating(text):
        return []
    counts = srec.get("counts") or {}
    target = normalize_label(target_label)
    n_dist = sum(int(counts.get(c, 0)) for c in MAIN_CLASSES if c != target)
    n_targ = int(counts.get(target, 0))
    out: list[str] = []
    if n_dist > 0:
        out.append("distractor")
    if n_targ > 0:
        out.append("target_attr")
    return out


# ── loaders ────────────────────────────────────────────────────────────────
def model_paths(model: str, parser_root: Path, traces_root: Path) -> tuple[Path, Path]:
    """(deepseek_sentence_dir, traces_dir) for the given model."""
    if model not in SUPPORTED_MODELS:
        raise SystemExit(f"unsupported model: {model}  (choose from {SUPPORTED_MODELS})")
    return parser_root / f"{model}_deepseek-sentence", traces_root / model


def load_sentences_per_image(parser_dir: Path, file_glob: str) -> dict[str, list[dict]]:
    per_image: dict[str, list[dict]] = defaultdict(list)
    files = sorted(parser_dir.glob(file_glob))
    print(f"[deepseek-sent] {len(files)} files")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                per_image[rec.get("image", "")].append(rec)
    print(f"[deepseek-sent] {len(per_image)} images, "
          f"{sum(len(v) for v in per_image.values())} sentences")
    return per_image


def load_original_traces(traces_root: Path) -> dict[str, dict]:
    """{image: {answer, label, n}} from <root>/<class>/<n>/*-conflicting.jsonl."""
    out: dict[str, dict] = {}
    n_files = 0
    for class_dir in sorted(traces_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for n_dir in sorted(class_dir.iterdir()):
            if not n_dir.is_dir():
                continue
            try:
                n_val = int(n_dir.name)
            except ValueError:
                continue
            for fp in n_dir.glob("*-conflicting.jsonl"):
                n_files += 1
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        img = rec.get("image", "")
                        if not img:
                            continue
                        out[img] = {
                            "answer": rec.get("answer", ""),
                            "label": normalize_label(rec.get("label", "")),
                            "n": int(rec.get("n_objects", n_val) or n_val),
                        }
    print(f"[traces] {n_files} files, {len(out)} images")
    return out


# ── text locating ──────────────────────────────────────────────────────────
def find_in_text_ws(haystack: str, needle: str):
    """Whitespace-tolerant substring search → (start, end) or None."""
    n = needle.strip()
    if not n:
        return None
    idx = haystack.find(n)
    if idx != -1:
        return (idx, idx + len(n))
    compact = " ".join(n.split())
    idx = haystack.find(compact)
    if idx != -1:
        return (idx, idx + len(compact))
    parts = compact.split(" ")
    pattern = r"\s+".join(re.escape(p) for p in parts if p)
    if pattern:
        m = re.search(pattern, haystack)
        if m:
            return (m.start(), m.end())
    first_line = n.splitlines()[0].strip()
    if first_line and first_line != n and len(first_line) >= 15:
        idx = haystack.find(first_line)
        if idx != -1:
            return (idx, idx + len(first_line))
        fl_compact = " ".join(first_line.split())
        fl_parts = fl_compact.split(" ")
        fl_pattern = r"\s+".join(re.escape(p) for p in fl_parts if p)
        if fl_pattern:
            m = re.search(fl_pattern, haystack)
            if m:
                return (m.start(), m.end())
    return None


def get_pre_answer_and_split(original: str) -> tuple[str, int, int]:
    """pre_answer prefix + (think_end, summary_end) char offsets within it.

    Cut point (where force-answer generation starts):
      - GLM-style outputs (contain ``<|begin_of_box|>``): right after that token.
      - Others (qwen3-thinking / intern / onevision): right after ``<answer>``.
    think_end   : start of ``\\n</think>`` inside the prefix (len(prefix) if absent).
    summary_end : len(prefix).
    """
    m_box = BEGIN_BOX_RE.search(original)
    if m_box is not None:
        pre_answer = original[:m_box.end()]
    else:
        m_ans = ANS_OPEN_RE.search(original)
        if m_ans is not None:
            pre_answer = original[:m_ans.end()]
        else:
            pre_answer = original.rstrip() + "\n\n<answer>"

    m_think = THINK_CLOSE_RE.search(pre_answer)
    think_end = m_think.start() if m_think is not None else len(pre_answer)
    summary_end = len(pre_answer)
    return pre_answer, think_end, summary_end


def collect_phrase_positions(prefix: str, sent_records: list, target: str, stype: str):
    """[(start, end, phrase, src_class)] sorted by start, unique by (start, end).

    Phrases are collected only inside sentences that qualify for ``stype``
    (see classify_sentence); every occurrence is a separate entry.
    """
    out = []
    for srec in sent_records:
        text = (srec.get("answer", "") or "").strip()
        if not text or is_prompt_restating(text):
            continue
        cls_list = classify_sentence(srec, target)
        if stype not in cls_list:
            continue
        sent_loc = find_in_text_ws(prefix, text)
        if sent_loc is None:
            continue
        sent_start, sent_end = sent_loc
        sent_span = prefix[sent_start:sent_end]
        classes = ([c for c in MAIN_CLASSES if c != target]
                   if stype == "distractor" else [target])
        for cls in classes:
            for phrase in srec.get(f"{cls}_attributes", []) or []:
                phrase = (phrase or "").strip()
                if not phrase:
                    continue
                search_start = 0
                while True:
                    sub = sent_span[search_start:]
                    loc = find_in_text_ws(sub, phrase)
                    if loc is None:
                        break
                    local_s, local_e = loc
                    abs_s = sent_start + search_start + local_s
                    abs_e = sent_start + search_start + local_e
                    out.append((abs_s, abs_e, phrase, cls))
                    search_start = search_start + local_e
                    if local_e == local_s:
                        break
    seen = set()
    dedup = []
    for s, e, ph, cls in sorted(out):
        key = (s, e)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((s, e, ph, cls))
    return dedup


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-thinking", choices=SUPPORTED_MODELS)
    ap.add_argument("--filter-n", type=int, default=4)
    ap.add_argument("--parser-root", type=Path, required=True,
                    help="DeepSeek sentence-level outputs: <root>/<model>_deepseek-sentence/*conflicting*.consolidated.jsonl")
    ap.add_argument("--traces-root", type=Path, required=True,
                    help="Idis-perception inference outputs: <root>/<model>/<class>/<n>/*-conflicting.jsonl")
    ap.add_argument("--output", type=Path, default=None,
                    help="default: out/inputs_<model>.parquet next to this script")
    args = ap.parse_args()

    if args.output is None:
        args.output = Path(__file__).resolve().parent / "out" / f"inputs_{args.model}.parquet"

    deepseek_dir, traces_dir = model_paths(args.model, args.parser_root, args.traces_root)
    print(f"▶ model={args.model}")
    print(f"  deepseek-sentence: {deepseek_dir}")
    print(f"  traces:            {traces_dir}")

    sents = load_sentences_per_image(deepseek_dir, "*conflicting*.consolidated.jsonl")
    origs = load_original_traces(traces_dir)
    common = [im for im in sorted(set(sents) & set(origs))
              if origs[im]["n"] == args.filter_n]
    print(f"common images (n={args.filter_n}): {len(common)}")

    rows = []
    for img in common:
        rec = origs[img]
        label = normalize_label(rec["label"])
        original = rec["answer"]
        if not original:
            continue
        pre_answer, think_end, summary_end = get_pre_answer_and_split(original)
        dist_spans = collect_phrase_positions(pre_answer, sents[img], label, "distractor")
        tgt_spans = collect_phrase_positions(pre_answer, sents[img], label, "target_attr")
        rows.append({
            "image": img,
            "label": label,
            "n": rec["n"],
            "pre_answer": pre_answer,
            "think_end_char": int(think_end),
            "summary_end_char": int(summary_end),
            "dist_spans": [(int(s), int(e)) for (s, e, *_rest) in dist_spans],
            "tgt_spans": [(int(s), int(e)) for (s, e, *_rest) in tgt_spans],
            "n_dist_spans": len(dist_spans),
            "n_tgt_spans": len(tgt_spans),
            "original_answer": original,
        })

    df = pd.DataFrame(rows)
    print("\n=== summary ===")
    print(f"  total images:                {len(df)}")
    print(f"  avg n_dist_spans / image:    {df['n_dist_spans'].mean():.2f}")
    print(f"  avg n_tgt_spans / image:     {df['n_tgt_spans'].mean():.2f}")
    print(f"  images with 0 dist_spans:    {(df['n_dist_spans']==0).sum()}")
    print(f"  images with 0 tgt_spans:     {(df['n_tgt_spans']==0).sum()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\n✅ saved: {args.output}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
