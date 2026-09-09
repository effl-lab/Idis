#!/usr/bin/env python3
"""Question files for the Idis-perception splits, read by inference/run_perception.py.

  input   : the dataset root written by gemini_edit.py, plus original/ for the no-distractor baseline
  output  : <out-dir>/<class>-<n>-<semantic>.jsonl, and <class>-original.jsonl with --include-original
  record  : {"image": <path relative to the dataset root>, "label": <short class>, "objects": <n>}
  paths   : relative, so run_perception.py resolves them against --image-folder <root>
  usage   : python build_question_files.py --image-root /path/to/idis_perception --include-original
"""

import argparse
import json
import os

from objects import SEMANTICS, class_from_dir, short_name

RESERVED_DIRS = {"original", "meta"}
DISTRACTOR_EXTS = (".png",)
ORIGINAL_EXTS = (".jpeg", ".jpg", ".png")


def list_images(directory, exts):
    if not os.path.isdir(directory):
        return []
    return [f for f in sorted(os.listdir(directory)) if f.lower().endswith(exts)]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{os.path.basename(path)}: {len(records)} images")


def class_dirs(image_root):
    """Class directories at the dataset root, e.g. `00_dog`, skipping original/ and meta/."""
    return [
        d for d in sorted(os.listdir(image_root))
        if d not in RESERVED_DIRS and os.path.isdir(os.path.join(image_root, d))
    ]


def build_distractor_files(image_root, out_dir, counts):
    total = 0
    for class_dir in class_dirs(image_root):
        label = short_name(class_from_dir(class_dir))
        for n in counts:
            for semantic in SEMANTICS:
                rel_dir = os.path.join(class_dir, str(n), semantic)
                images = list_images(os.path.join(image_root, rel_dir), DISTRACTOR_EXTS)
                if not images:
                    continue
                records = [
                    {"image": os.path.join(rel_dir, name), "label": label, "objects": n}
                    for name in images
                ]
                write_jsonl(os.path.join(out_dir, f"{label}-{n}-{semantic}.jsonl"), records)
                total += len(records)
    return total


def build_original_files(image_root, out_dir):
    original_root = os.path.join(image_root, "original")
    if not os.path.isdir(original_root):
        print(f"No original/ under {image_root}, skipping the baseline files")
        return 0
    total = 0
    for class_dir in sorted(os.listdir(original_root)):
        rel_dir = os.path.join("original", class_dir)
        images = list_images(os.path.join(image_root, rel_dir), ORIGINAL_EXTS)
        if not images:
            continue
        label = short_name(class_from_dir(class_dir))
        records = [
            {"image": os.path.join(rel_dir, name), "label": label, "objects": 0}
            for name in images
        ]
        write_jsonl(os.path.join(out_dir, f"{label}-original.jsonl"), records)
        total += len(records)
    return total


def main():
    ap = argparse.ArgumentParser(description="Build run_perception.py question files from the Idis-perception tree")
    ap.add_argument("--image-root", required=True, help="dataset root holding the class directories")
    ap.add_argument("--out-dir", default=None, help="output directory (default: <image-root>/meta)")
    ap.add_argument("--counts", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
                    help="distractor counts to emit")
    ap.add_argument("--include-original", action="store_true",
                    help="also emit <class>-original.jsonl for the no-distractor baseline")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.image_root, "meta")
    os.makedirs(out_dir, exist_ok=True)

    total = build_distractor_files(args.image_root, out_dir, args.counts)
    if args.include_original:
        total += build_original_files(args.image_root, out_dir)
    print(f"Total {total} images written to {out_dir}")


if __name__ == "__main__":
    main()
