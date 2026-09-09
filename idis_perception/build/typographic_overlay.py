"""Typographic distractors for Idis-perception: render a non-target class name into each image (App. A.1).

  input  : <image-root>/original/<class dir>/<stem>.JPEG  (ImageNet-9 original split, 9 x 450)
  output : <image-root>/typographic/<class dir>/<overlay>/<stem>.png  - one per non-target class (8 per image)
           <image-root>/meta/<class>-typographic.jsonl   {"image", "label", "overlay_class"} for run_perception.py
  style  : bottom-left, bold text (7% of image width) on a rounded semi-transparent box
  usage  : python typographic_overlay.py --image-root /path/to/idis_perception
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from objects import OBJECTS, class_from_dir, short_name

CLASSES = [short_name(c) for c in OBJECTS]          # dog, bird, vehicle, ... fish (overlay text)
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}

# Overlay style (fixed, reproduces the released images).
TEXT_SCALE = 0.07      # font size = 7% of image width
MARGIN = (20, 20)      # box offset from the bottom-left corner
BOX_PAD = (10, 8)
BOX_ALPHA = 140
ROUND_R = 8
STROKE_W = 2


def default_font() -> str:
    """DejaVuSans-Bold shipped with matplotlib (no system font dependency)."""
    import matplotlib
    return os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")


def draw_label(img: Image.Image, text: str, font_path: str) -> Image.Image:
    W, H = img.size
    font = ImageFont.truetype(font_path, max(12, int(W * TEXT_SCALE)))

    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font, stroke_width=STROKE_W)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = BOX_PAD
    box_w, box_h = text_w + 2 * pad_x, text_h + 2 * pad_y
    box_x, box_y = MARGIN[0], H - MARGIN[1] - box_h

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h], radius=ROUND_R, fill=(0, 0, 0, BOX_ALPHA)
    )
    composed = Image.alpha_composite(base, overlay)
    ImageDraw.Draw(composed).text(
        (box_x + pad_x - bbox[0], box_y + pad_y - bbox[1]), text, font=font,
        fill=(255, 255, 255, 255), stroke_width=STROKE_W, stroke_fill=(0, 0, 0, 255),
    )
    return composed.convert("RGB")


def iter_sources(source_root: Path):
    for cls_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        label = short_name(class_from_dir(cls_dir.name))
        if label not in CLASSES:
            print(f"[warn] unknown class dir, skipped: {cls_dir}", file=sys.stderr)
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                yield cls_dir.name, label, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-root", type=Path, required=True, help="dataset root holding original/<class dir>/")
    ap.add_argument("--source-dir", default="original", help="subdirectory with the ImageNet-9 class directories")
    ap.add_argument("--out-dir", default="typographic", help="output subdirectory under image-root")
    ap.add_argument("--meta-dir", default="meta", help="question-file subdirectory under image-root")
    ap.add_argument("--font", default=None, help="TTF path (default: matplotlib's DejaVuSans-Bold)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process at most N source images (0 = all)")
    args = ap.parse_args()

    font_path = args.font or default_font()
    source_root = args.image_root / args.source_dir
    if not source_root.is_dir():
        raise SystemExit(f"missing source directory: {source_root}")
    meta_root = args.image_root / args.meta_dir
    meta_root.mkdir(parents=True, exist_ok=True)

    sources = list(iter_sources(source_root))
    if args.limit:
        sources = sources[: args.limit]
    print(f"sources: {len(sources)} → outputs: {len(sources) * (len(CLASSES) - 1)}")

    n_ok = n_skip = n_err = 0
    records = {}  # label → list of rows
    for i, (dir_name, label, src) in enumerate(sources, 1):
        try:
            img = Image.open(src)
            img.load()
        except Exception as e:
            n_err += 1
            print(f"[err] open {src}: {e}", file=sys.stderr)
            continue
        for overlay in CLASSES:
            if overlay == label:
                continue
            rel = Path(args.out_dir) / dir_name / overlay / f"{src.stem}.png"
            dst = args.image_root / rel
            if dst.exists() and not args.overwrite:
                n_skip += 1
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    draw_label(img, overlay, font_path).save(dst)
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    print(f"[err] render {src} ({overlay}): {e}", file=sys.stderr)
                    continue
            records.setdefault(label, []).append({"image": str(rel), "label": label, "overlay_class": overlay})
        if i % 100 == 0:
            print(f"  {i}/{len(sources)}  rendered={n_ok} skipped={n_skip} err={n_err}", flush=True)

    for label, rows in records.items():
        path = meta_root / f"{label}-typographic.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(rows)} images")
    print(f"done: rendered={n_ok} skipped={n_skip} err={n_err} → {args.image_root / args.out_dir}")


if __name__ == "__main__":
    main()
