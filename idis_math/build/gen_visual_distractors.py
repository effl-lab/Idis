#!/usr/bin/env python3
"""Aligned / conflicting visual distractors for Idis-math.

  layout      : [original | 2x2 grid of synthetic shapes], n1..n4 cells filled
  aligned     : same coarse concept family as the diagram
  conflicting : different concept family
  labels      : data/idis_math/shape_labels/
  usage       : python gen_visual_distractors.py --mode {aligned,conflicting,both}
"""

import argparse
import json
import os
import random

from PIL import Image
from tqdm import tqdm

# Disable PIL decompression bomb warning for our own safe concat outputs
Image.MAX_IMAGE_PIXELS = None

from draw_shapes import draw_coarse, get_specific_shape

# ── Paths ──────────────────────────────────────────────────────────

# Paths are set in main() via configure_paths().
TESTMINI = IMG_ROOT = AUG_ROOT = OUT_ROOT = None
OUT_DIR_NAME = "library"
LABEL_FILES = {}
SUBJECT_LABEL_FILES = {
    "Plane Geometry": "shape_labels_final_plane_geometry.jsonl",
    "Solid Geometry": "shape_labels_final_solid_geometry.jsonl",
    "Functions": "shape_labels_final_functions.jsonl",
}
DEFAULT_LABELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "idis_math", "shape_labels")


def configure_paths(mathverse_root, aug_root=None, labels_dir=None, out_name="library"):
    """mathverse_root holds testmini.json and testmini/images/; outputs go to aug_root/out_name."""
    global TESTMINI, IMG_ROOT, AUG_ROOT, OUT_ROOT, OUT_DIR_NAME, LABEL_FILES
    TESTMINI = os.path.join(mathverse_root, "testmini.json")
    IMG_ROOT = os.path.join(mathverse_root, "testmini", "images")
    AUG_ROOT = aug_root or os.path.join(mathverse_root, "mathverse_aug")
    OUT_DIR_NAME = out_name
    OUT_ROOT = os.path.join(AUG_ROOT, OUT_DIR_NAME)
    labels_dir = labels_dir or DEFAULT_LABELS_DIR
    LABEL_FILES = {subj: os.path.join(labels_dir, fn) for subj, fn in SUBJECT_LABEL_FILES.items()}


TARGET_SUBJECTS = {"Plane Geometry", "Solid Geometry", "Functions"}
SKIP_VERSIONS = {"Text Only"}

NUM_DISTRACTORS = 4  # n1~n4, always 4 cells in 2x2 grid

# ── Conflict maps ─────────────────────────────────────────────────

PLANE_CONFLICT_POOL = ["triangle", "quadrilateral", "circle_family"]
SOLID_CONFLICT_POOL = [
    "prism_family", "pyramid_family", "cylinder_family",
    "cone_family", "sphere_family",
]
ALIGNED_SKIP_LABELS = {"other", "other_2d_or_ambiguous"}


# ── Helpers ────────────────────────────────────────────────────────

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def load_label_map():
    label_map = {}
    for subject, path in LABEL_FILES.items():
        if not os.path.exists(path):
            continue
        for item in load_jsonl(path):
            pid = str(item["problem_index"])
            label_map[(subject, pid)] = item["final_coarse"]
    return label_map


def load_target_samples():
    with open(TESTMINI, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for d in data:
        subject = d["metadata"]["subject"]
        if subject not in TARGET_SUBJECTS:
            continue
        if d["problem_version"] in SKIP_VERSIONS:
            continue
        samples.append({
            "problem_index": str(d["problem_index"]),
            "sample_index": str(d["sample_index"]),
            "subject": subject,
            "problem_version": d["problem_version"],
            "image": d["image"],
        })
    return samples


def concat_grid(orig: Image.Image, distractors: list, n_filled: int) -> Image.Image:
    """Place distractors in 2x2 grid to the right of original.

    Layout (H = orig.height, cell = H//2):
        ┌─────────┬──────┬──────┐
        │         │  d1  │  d2  │
        │ original├──────┼──────┤
        │         │  d3  │  d4  │
        └─────────┴──────┴──────┘

    - Original pasted unchanged at (0, 0)
    - Right panel: 2x2 grid, each cell cell x cell
    - First n_filled distractors placed (row-major)
    - Empty cells remain white
    """
    H = orig.height
    cell = H // 2
    panel_w = cell * 2  # ≈ H
    total_w = orig.width + panel_w

    out = Image.new("RGB", (total_w, H), "white")
    out.paste(orig, (0, 0))  # preserve original pixel-for-pixel

    positions = [
        (orig.width, 0),               # d1: top-left
        (orig.width + cell, 0),        # d2: top-right
        (orig.width, cell),            # d3: bottom-left
        (orig.width + cell, cell),     # d4: bottom-right
    ]

    for i in range(min(n_filled, 4, len(distractors))):
        d = distractors[i]
        if d.size != (cell, cell):
            d = d.resize((cell, cell), Image.LANCZOS)
        out.paste(d, positions[i])

    return out


def get_conflict_pool(subject):
    if subject == "Solid Geometry":
        return SOLID_CONFLICT_POOL
    return PLANE_CONFLICT_POOL


def get_aligned_labels(final_coarse):
    """Return valid labels to generate aligned sets for.

    - Skip other / other_2d_or_ambiguous
    - Always ignore line_angle_configuration
    - If nothing valid remains, return empty list
    """
    valid = []
    for label in final_coarse:
        if label in ALIGNED_SKIP_LABELS:
            continue
        if label == "line_angle_configuration":
            continue
        valid.append(label)
    return valid


def get_conflicting_labels(final_coarse, pool):
    """Return conflict labels to generate sets for.

    - Exclude all labels in final_coarse from the pool
    - If final_coarse is just [other], use full pool
    """
    fc_set = set(final_coarse)
    remaining = [c for c in pool if c not in fc_set]
    return remaining


# ── Generation ─────────────────────────────────────────────────────

def generate_aligned(samples, label_map, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.jsonl")
    meta_f = open(meta_path, "w", encoding="utf-8")
    count = 0

    for s in tqdm(samples, desc="aligned"):
        subject = s["subject"]
        if subject == "Functions":
            continue

        pid = s["problem_index"]
        final_coarse = label_map.get((subject, pid))
        if final_coarse is None:
            continue

        aligned_labels = get_aligned_labels(final_coarse)
        if not aligned_labels:
            continue

        img_path = os.path.join(IMG_ROOT, s["image"])
        if not os.path.exists(img_path):
            continue
        try:
            orig = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        cell = orig.height // 2
        if cell < 16:
            continue  # too small to render distractors

        for label in aligned_labels:
            rng = random.Random(seed + int(pid) + hash(label) + hash(s["image"]))

            # Generate 4 distractors of the same coarse label
            distractor_imgs = [draw_coarse(label, cell, rng) for _ in range(NUM_DISTRACTORS)]

            augmented = {}
            for k in range(1, NUM_DISTRACTORS + 1):
                out_img = concat_grid(orig, distractor_imgs, k)
                rel_path = f"{OUT_DIR_NAME}/aligned/n{k}/{label}/{s['image']}"
                dst = os.path.join(AUG_ROOT, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                out_img.save(dst)
                augmented[f"n{k}"] = rel_path

            meta_f.write(json.dumps({
                "problem_index": pid,
                "sample_index": s["sample_index"],
                "subject": subject,
                "problem_version": s["problem_version"],
                "original_labels": final_coarse,
                "primary_label": label,
                "distractor_labels": [label] * NUM_DISTRACTORS,
                "original": s["image"],
                "augmented": augmented,
            }, ensure_ascii=False) + "\n")
            meta_f.flush()
            count += 1

    meta_f.close()
    print(f"Aligned: {count} sets generated -> {out_dir}")


def generate_conflicting(samples, label_map, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.jsonl")
    meta_f = open(meta_path, "w", encoding="utf-8")
    count = 0

    for s in tqdm(samples, desc="conflicting"):
        subject = s["subject"]
        pid = s["problem_index"]
        final_coarse = label_map.get((subject, pid))
        if final_coarse is None:
            continue

        pool = get_conflict_pool(subject)
        candidates = get_conflicting_labels(final_coarse, pool)
        if not candidates:
            continue

        img_path = os.path.join(IMG_ROOT, s["image"])
        if not os.path.exists(img_path):
            continue
        try:
            orig = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        cell = orig.height // 2
        if cell < 16:
            continue

        # Generate one set per candidate conflict label
        for conflict_label in candidates:
            rng = random.Random(seed + int(pid) + hash(s["image"]) + hash(conflict_label))

            distractor_imgs = [draw_coarse(conflict_label, cell, rng) for _ in range(NUM_DISTRACTORS)]

            augmented = {}
            for k in range(1, NUM_DISTRACTORS + 1):
                out_img = concat_grid(orig, distractor_imgs, k)
                rel_path = f"{OUT_DIR_NAME}/conflicting/n{k}/{conflict_label}/{s['image']}"
                dst = os.path.join(AUG_ROOT, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                out_img.save(dst)
                augmented[f"n{k}"] = rel_path

            meta_f.write(json.dumps({
                "problem_index": pid,
                "sample_index": s["sample_index"],
                "subject": subject,
                "problem_version": s["problem_version"],
                "original_labels": final_coarse,
                "conflict_label": conflict_label,
                "distractor_labels": [conflict_label] * NUM_DISTRACTORS,
                "original": s["image"],
                "augmented": augmented,
            }, ensure_ascii=False) + "\n")
            meta_f.flush()
            count += 1

    meta_f.close()
    print(f"Conflicting: {count} sets generated -> {out_dir}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["aligned", "conflicting", "both"],
                    default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mathverse-root", default=os.environ.get("MATHVERSE_ROOT"),
                    help="directory with testmini.json and testmini/images/ (env: MATHVERSE_ROOT)")
    ap.add_argument("--aug-root", default=None, help="output root (default: <mathverse-root>/mathverse_aug)")
    ap.add_argument("--labels-dir", default=None, help="shape label JSONLs (default: data/idis_math/shape_labels)")
    ap.add_argument("--out-name", default="library", help="output subdirectory name under aug-root")
    args = ap.parse_args()
    if not args.mathverse_root:
        ap.error("--mathverse-root (or MATHVERSE_ROOT) is required")
    configure_paths(args.mathverse_root, args.aug_root, args.labels_dir, args.out_name)

    label_map = load_label_map()
    print(f"Loaded {len(label_map)} shape label entries")

    samples = load_target_samples()
    print(f"Target samples: {len(samples)} (excl Text Only)")

    from collections import Counter
    subj_counts = Counter(s["subject"] for s in samples)
    for subj, cnt in subj_counts.most_common():
        print(f"  {subj}: {cnt}")

    if args.mode in ("aligned", "both"):
        out_dir = os.path.join(OUT_ROOT, "aligned")
        generate_aligned(samples, label_map, args.seed, out_dir)

    if args.mode in ("conflicting", "both"):
        out_dir = os.path.join(OUT_ROOT, "conflicting")
        generate_conflicting(samples, label_map, args.seed, out_dir)


if __name__ == "__main__":
    main()
