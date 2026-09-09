#!/usr/bin/env python3
"""Irrelevant visual distractors for Idis-math (LogicVista table images).

  layout : [original | 2x2 grid], n1..n4 cells filled
  output : <aug-root>/library/irrelevant/n{k}/ + meta.jsonl
"""

import argparse
import json
import os
import random

from PIL import Image
from tqdm import tqdm

# Avoid decompression bomb warnings for our safe outputs
Image.MAX_IMAGE_PIXELS = None

# ── Paths ──────────────────────────────────────────────────────────

# Paths are set in main() via configure_paths().
TESTMINI = IMG_ROOT = AUG_ROOT = OUT_ROOT = None
OUT_DIR_NAME = "library"
LOGICVISTA_ROOT = LOGICVISTA_META = None


def configure_paths(mathverse_root, logicvista_root, aug_root=None, out_name="library"):
    """mathverse_root holds testmini.json and testmini/images/; logicvista_root holds metadata.json."""
    global TESTMINI, IMG_ROOT, AUG_ROOT, OUT_ROOT, OUT_DIR_NAME, LOGICVISTA_ROOT, LOGICVISTA_META
    TESTMINI = os.path.join(mathverse_root, "testmini.json")
    IMG_ROOT = os.path.join(mathverse_root, "testmini", "images")
    AUG_ROOT = aug_root or os.path.join(mathverse_root, "mathverse_aug")
    OUT_DIR_NAME = out_name
    OUT_ROOT = os.path.join(AUG_ROOT, OUT_DIR_NAME)
    LOGICVISTA_ROOT = logicvista_root
    LOGICVISTA_META = os.path.join(logicvista_root, "metadata.json")


TARGET_SUBJECTS = {"Plane Geometry", "Solid Geometry", "Functions"}
SKIP_VERSIONS = {"Text Only"}
NUM_DISTRACTORS = 4



# ── Helpers ────────────────────────────────────────────────────────

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
    """Place distractors in a 2x2 grid to the right of the original (same layout as gen_visual_distractors.py)."""
    H = orig.height
    cell = H // 2
    panel_w = cell * 2
    total_w = orig.width + panel_w

    out = Image.new("RGB", (total_w, H), "white")
    out.paste(orig, (0, 0))

    positions = [
        (orig.width, 0),
        (orig.width + cell, 0),
        (orig.width, cell),
        (orig.width + cell, cell),
    ]

    for i in range(min(n_filled, 4, len(distractors))):
        d = distractors[i]
        if d.size != (cell, cell):
            d = d.resize((cell, cell), Image.LANCZOS)
        out.paste(d, positions[i])

    return out




# ── Pool loaders ───────────────────────────────────────────────────

def load_logicvista_pool():
    with open(LOGICVISTA_META, "r", encoding="utf-8") as f:
        data = json.load(f)
    pool = []
    for item in data:
        if "table" not in item.get("specific_capability", []):
            continue
        vid = item.get("id", "")
        if not vid.startswith("v1_"):
            continue
        x = int(vid.split("_")[1])
        if x % 3 == 0:
            img_path = os.path.join(LOGICVISTA_ROOT, item["image"])
            if os.path.exists(img_path):
                pool.append({"id": vid, "path": img_path})
    return pool




# ── Generation ─────────────────────────────────────────────────────

def generate_variant(variant, samples, pool, seed, loader_fn):
    """Generate n1..n4 grid compositions for one distractor variant.

    Args:
        variant: "irrelevant"
        samples: target samples
        pool: list of distractor sources (paths or dicts)
        loader_fn: callable(pool_item) -> PIL.Image
    """
    out_dir = os.path.join(OUT_ROOT, variant)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.jsonl")
    meta_f = open(meta_path, "w", encoding="utf-8")
    count = 0

    rng = random.Random(seed)

    for s in tqdm(samples, desc=variant):
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

        # Pick 4 distractors (with replacement for small pools)
        if len(pool) >= NUM_DISTRACTORS:
            picks = rng.sample(pool, NUM_DISTRACTORS)
        else:
            picks = [rng.choice(pool) for _ in range(NUM_DISTRACTORS)]

        distractor_imgs = []
        distractor_info = []
        for p in picks:
            try:
                d_img = loader_fn(p)
                distractor_imgs.append(d_img)
                # Extract id/filename
                if isinstance(p, dict):
                    distractor_info.append({"id": p.get("id", "")})
                else:
                    distractor_info.append(os.path.basename(p))
            except Exception as e:
                print(f"  [skip distractor] {p}: {e}")

        if len(distractor_imgs) < NUM_DISTRACTORS:
            continue

        augmented = {}
        for k in range(1, NUM_DISTRACTORS + 1):
            out_img = concat_grid(orig, distractor_imgs, k)
            rel_path = f"{OUT_DIR_NAME}/{variant}/n{k}/{s['image']}"
            dst = os.path.join(AUG_ROOT, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            out_img.save(dst)
            augmented[f"n{k}"] = rel_path

        meta_f.write(json.dumps({
            "problem_index": s["problem_index"],
            "sample_index": s["sample_index"],
            "subject": s["subject"],
            "problem_version": s["problem_version"],
            "original": s["image"],
            "distractors": distractor_info,
            "augmented": augmented,
        }, ensure_ascii=False) + "\n")
        meta_f.flush()
        count += 1

    meta_f.close()
    print(f"{variant}: {count} sets generated -> {out_dir}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mathverse-root", default=os.environ.get("MATHVERSE_ROOT"),
                    help="directory with testmini.json and testmini/images/ (env: MATHVERSE_ROOT)")
    ap.add_argument("--logicvista-root", default=os.environ.get("LOGICVISTA_ROOT"),
                    help="LogicVista export directory with metadata.json (env: LOGICVISTA_ROOT)")
    ap.add_argument("--aug-root", default=None, help="output root (default: <mathverse-root>/mathverse_aug)")
    ap.add_argument("--out-name", default="library", help="output subdirectory name under aug-root")
    args = ap.parse_args()
    if not args.mathverse_root or not args.logicvista_root:
        ap.error("--mathverse-root and --logicvista-root (or MATHVERSE_ROOT / LOGICVISTA_ROOT) are required")
    configure_paths(args.mathverse_root, args.logicvista_root, args.aug_root, args.out_name)

    samples = load_target_samples()
    print(f"Target samples: {len(samples)} (excl Text Only)")

    pool = load_logicvista_pool()
    print(f"Irrelevant pool: {len(pool)} images")
    generate_variant("irrelevant", samples, pool, args.seed,
                     loader_fn=lambda p: Image.open(p["path"]).convert("RGB"))



if __name__ == "__main__":
    main()
