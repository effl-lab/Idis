"""Accuracy tables from run_block_force.py outputs.

  arm    ∈ {baseline, block_distractor, block_target}
  layer  ∈ {all, mid_late}
  output : accuracy by (model, arm, layer), per class, and Δ vs baseline
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

MODELS = ["qwen3-thinking", "glm", "onevision", "intern"]
LABEL_ALIASES = {"wheeled vehicle": "vehicle", "musical instrument": "instrument"}
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|$)", re.IGNORECASE | re.DOTALL)
# strip GLM-style boxed control tokens and partial closing tags
SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PARTIAL_TAG_RE = re.compile(r"</[a-z]*$", re.IGNORECASE)


def normalize_label(s: str) -> str:
    s = (s or "").strip().lower().replace("_", " ")
    return LABEL_ALIASES.get(s, s)


def extract_pred(continuation: str) -> str | None:
    """v2 continuations are short and often end mid-tag (`Dog</`, `Dog</answer`,
    `Dog<|end_of_box|>.</answer>`). Strip partials, take leading words."""
    if not continuation:
        return None
    text = continuation.strip()
    # If there's an explicit <answer>...</answer> we honour it; else take everything
    m = ANSWER_RE.search(text)
    cand = m.group(1) if m else text
    cand = cand.split("</answer>")[0]
    cand = SPECIAL_RE.sub("", cand)
    cand = PARTIAL_TAG_RE.sub("", cand)        # drop trailing `</`, `</answer`
    cand = cand.strip().strip("`'\"*<>").lower()
    cand = re.sub(r"[.,;:!?\s]+$", "", cand)
    cand = cand.replace("_", " ").strip()
    if not cand:
        return None
    return normalize_label(cand)


def load_model(model: str, root: Path) -> pd.DataFrame:
    """Load every run_block_force.py JSONL under <root>/<model>/."""
    rows: list[dict] = []
    for fp in sorted((root / model).rglob("*.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["model"] = model
                r["class"] = normalize_label(r.get("label", ""))
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    df["pred"] = df["continuation"].map(extract_pred)
    df["gold"] = df["label"].map(normalize_label)
    df["correct"] = (df["pred"] == df["gold"]).astype(int)
    df["unparseable"] = df["pred"].isna().astype(int)
    df["layer"] = df["layer"].fillna("-")
    return df


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False).agg(
        n=("image", "size"),
        n_imgs=("image", "nunique"),
        acc=("correct", "mean"),
        n_correct=("correct", "sum"),
        n_unparseable=("unparseable", "sum"),
    ).round(4).reset_index()
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="directory containing <model>/*.jsonl from run_block_force.py")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <root>/summary")
    args = ap.parse_args()
    out_dir = args.out_dir or (args.root / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for m in MODELS:
        df = load_model(m, args.root)
        if df.empty:
            print(f"[skip] {m}: empty")
            continue
        all_dfs.append(df)
        print(f"[load] {m:<16}  records={len(df):>6}  images={df['image'].nunique():>4}  "
              f"classes={df['class'].nunique()}")
    if not all_dfs:
        raise SystemExit("no data")
    df = pd.concat(all_dfs, ignore_index=True)

    # 1) overall (model × arm × layer)
    overall = summarize(df, ["model", "arm", "layer"])
    overall.to_csv(out_dir / "accuracy_by_model_arm_layer.csv",
                   index=False, float_format="%.5f")

    # 2) per-class (model × class × arm × layer)
    per_cls = summarize(df, ["model", "class", "arm", "layer"])
    per_cls.to_csv(out_dir / "accuracy_by_model_class_arm_layer.csv",
                   index=False, float_format="%.5f")

    # 3) pivot table: rows = (arm, layer), cols = model
    print("\n" + "=" * 80)
    print("ACCURACY by model × (arm, layer)   — pooled across 9 classes")
    print("=" * 80)
    pv = overall.pivot_table(index=["arm", "layer"], columns="model", values="acc")
    # reorder models, fix arm order
    arm_order = ["baseline", "block_distractor", "block_target"]
    layer_order = ["-", "all", "mid_late"]
    pv = pv.reindex([(a, l) for a in arm_order for l in layer_order if (a, l) in pv.index])
    pv = pv.reindex(columns=[m for m in MODELS if m in pv.columns])
    print(pv.round(4).to_string())

    # Δ vs baseline
    print("\nΔ ACC vs baseline")
    base = pv.loc[("baseline", "-")] if ("baseline", "-") in pv.index else None
    if base is not None:
        delta = pv.subtract(base, axis=1)
        print(delta.round(4).to_string())
        delta.round(4).to_csv(out_dir / "delta_vs_baseline.csv", float_format="%.4f")
    pv.round(4).to_csv(out_dir / "accuracy_pivot.csv", float_format="%.4f")

    # 4) sanity: per-class baseline (model × class)
    print("\n" + "=" * 80)
    print("BASELINE accuracy per (model, class)")
    print("=" * 80)
    base_cls = per_cls[per_cls["arm"] == "baseline"].pivot_table(
        index="class", columns="model", values="acc")
    base_cls = base_cls.reindex(columns=[m for m in MODELS if m in base_cls.columns])
    print(base_cls.round(4).to_string())
    base_cls.round(4).to_csv(out_dir / "baseline_per_class.csv", float_format="%.4f")

    # 5) records / image counts sanity
    print("\n" + "=" * 80)
    print("RECORDS per (model, class)  — expect 9 cond × n_imgs")
    print("=" * 80)
    cnt = df.groupby(["model", "class"]).agg(
        records=("image", "size"),
        images=("image", "nunique"),
    ).reset_index()
    cnt["records_per_image"] = (cnt["records"] / cnt["images"]).round(2)
    pv_cnt = cnt.pivot_table(index="class", columns="model", values="records")
    pv_cnt = pv_cnt.reindex(columns=[m for m in MODELS if m in pv_cnt.columns])
    print(pv_cnt.fillna(0).astype(int).to_string())

    print(f"\n[saved] {out_dir}/")


if __name__ == "__main__":
    main()
