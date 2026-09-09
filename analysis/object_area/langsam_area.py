"""Target / distractor pixel areas with LangSAM (Sec. 6.1).

  input  : JSONL rows {image, label, objects}
  output : per-image meta JSON (fg_pixels, oth_pixels, ratios), masks, overlay
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import re
import difflib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lang_sam import LangSAM


# ----------------------------------------------------------------------
# Class → candidate objects
# ----------------------------------------------------------------------
CLASS_OBJECTS = {
    "dog": [
        "dog bone chew toy",
        "dog bowl",
        "tennis ball",
        "kennel",
    ],
    "bird": [
        "birdcage",
        "nest",
        "feather",
        "bird feeder",
    ],
    "wheeled vehicle": [
        "tire",
        "steering wheel",
        "license plate",
        "bumper",
    ],
    "reptile": [
        "terrarium rock",
        "heat lamp",
        "log hideout",
        "shed skin",
    ],
    "carnivore": [
        "toy fang",
        "blood stain",
        "meat",
        "skeletal animal carcass",
    ],
    "insect": [
        "trash bag",
        "empty net designed for catching insects",
        "fruit peel",
        "flower",
    ],
    "musical instrument": [
        "chalkboard",
        "music stand",
        "metronome",
        "sheet music",
    ],
    "primate": [
        "patch of jungle foliage",
        "banana",
        "coconut",
        "vine",
    ],
    "fish": [
        "fishing rod",
        "large empty nylon fishing net",
        "life jacket",
        "coral ornament",
    ],
}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def sanitize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.-]", "", s)
    return s


def parse_from_path(img_path: Path):
    """
    Expected: .../<class_dir>/<n_dir>/<case_dir>/<file>
    e.g., .../01_bird/1/conflicting/n01537544_29310.png
    """
    parts = img_path.parts
    if len(parts) < 4:
        return None, None, None

    class_dir, n_dir, case_dir = parts[-4], parts[-3], parts[-2]
    label = re.sub(r"^\d+[_-]?", "", class_dir).lower()

    try:
        n_objects = int(n_dir)
    except Exception:
        n_objects = None

    case = case_dir.lower()
    return label, n_objects, case


def to_numpy_mask(m):
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    return np.array(m, dtype=bool)


def save_overlay(
    image_pil: Image.Image,
    mask_bool: np.ndarray,
    out_png_path: Path,
    rgba=(0.0, 0.0, 0.0, 0.35),
    title: str | None = None,
) -> None:
    h, w = mask_bool.shape
    fig, ax = plt.subplots(1, figsize=(9, 9))
    ax.imshow(image_pil)
    ax.axis("off")

    overlay = np.zeros((h, w, 4), dtype=float)
    overlay[..., 0], overlay[..., 1], overlay[..., 2], overlay[..., 3] = rgba
    overlay[~mask_bool] = 0.0
    ax.imshow(overlay)

    if title:
        ax.set_title(title)

    fig.tight_layout(pad=0)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png_path), bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Conflicting class sweep
# ----------------------------------------------------------------------
def predict_conflict_class(model, image_pil: Image.Image, ground_label: str):
    candidates = [c for c in CLASS_OBJECTS.keys() if c != ground_label]

    class_found = {}
    class_area = {}
    class_unique = {}
    class_unique_ratio = {}
    sweep_prompts = {}
    sweep_obj_hits = {}
    first_obj_found = {}
    first_obj_hits = {}

    for cls in candidates:
        prompt = f"{ground_label}. " + ". ".join(CLASS_OBJECTS[cls]) + "."
        sweep_prompts[cls] = prompt

        pred = model.predict([image_pil], [prompt])[0]
        labels = [str(lb) for lb in pred.get("labels", [])]
        masks = [to_numpy_mask(m) for m in pred.get("masks", [])]

        obj_keys = [norm(o) for o in CLASS_OBJECTS[cls]]
        obj_hits_norm = {k: 0 for k in obj_keys}

        found_instances = 0
        area_sum = 0
        for lb, mk in zip(labels, masks):
            nlb = norm(lb)
            if nlb in obj_hits_norm:
                obj_hits_norm[nlb] += 1
                found_instances += 1
                area_sum += int(mk.sum())

        unique_detected = sum(1 for v in obj_hits_norm.values() if v > 0)
        ratio = (unique_detected / len(obj_hits_norm)) if obj_hits_norm else 0.0

        sweep_obj_hits[cls] = {o: obj_hits_norm[norm(o)] for o in CLASS_OBJECTS[cls]}
        class_found[cls] = int(found_instances)
        class_area[cls] = int(area_sum)
        class_unique[cls] = int(unique_detected)
        class_unique_ratio[cls] = float(ratio)

        first_obj = obj_keys[0] if obj_keys else None
        hits_first = obj_hits_norm.get(first_obj, 0) if first_obj is not None else 0
        first_obj_found[cls] = hits_first > 0
        first_obj_hits[cls] = int(hits_first)

    high_priority = [c for c in candidates if first_obj_found.get(c, False)]
    if high_priority:
        best = sorted(
            high_priority,
            key=lambda c: (
                first_obj_hits[c],
                class_unique[c],
                class_found[c],
                class_area[c],
            ),
            reverse=True,
        )[0]
    else:
        best = sorted(
            candidates,
            key=lambda c: (
                class_unique[c],
                class_found[c],
                class_area[c],
            ),
            reverse=True,
        )[0]

    return (
        best,
        class_found,
        class_area,
        sweep_prompts,
        class_unique,
        class_unique_ratio,
        sweep_obj_hits,
        first_obj_found,
        first_obj_hits,
    )


# ----------------------------------------------------------------------
# Label normalization
# ----------------------------------------------------------------------
def canonize_label(
    raw_label: str,
    allowed_targets: set[str],
    others_whitelist: set[str],
) -> str | None:
    t = norm(raw_label)
    candidates = list(allowed_targets | others_whitelist)

    if t in candidates:
        return t

    words = set(t.split())
    subs = []
    for w in candidates:
        wset = set(w.split())
        if wset.issubset(words):
            subs.append((len(wset), len(w), w))
    if subs:
        subs.sort(reverse=True)
        return subs[0][2]

    for w in candidates:
        if w in t:
            return w

    matches = difflib.get_close_matches(t, candidates, n=1, cutoff=0.8)
    return matches[0] if matches else None


# ----------------------------------------------------------------------
# One-step segmentation + saving
# ----------------------------------------------------------------------
def run_step_and_save(
    model,
    image_pil: Image.Image,
    prompt_text: str,
    step_idx: int,
    allowed_targets: set[str],
    others_whitelist: set[str],
    fg_rgba,
    oth_rgba,
    overlay_dir: Path,
    file_stem: str,
    parsed_n,
    out_case: str,
    save_npy: bool = True,
    mask_dir: Path | None = None,
    variant_tag: str | None = None,
) -> dict:
    width, height = image_pil.size
    pred = model.predict([image_pil], [prompt_text])[0]

    labels = [str(lb) for lb in pred.get("labels", [])]
    masks = list(pred.get("masks", []))
    if len(masks) != len(labels):
        min_len = min(len(masks), len(labels))
        labels = labels[:min_len]
        masks = masks[:min_len]

    masks_np = [to_numpy_mask(mk) for mk in masks]

    if not masks_np:
        fg_mask = np.zeros((height, width), dtype=bool)
        oth_mask = np.zeros((height, width), dtype=bool)
    else:
        fg_mask = np.zeros_like(masks_np[0], dtype=bool)
        oth_mask = np.zeros_like(masks_np[0], dtype=bool)
        for lb, mk in zip(labels, masks_np):
            cname = canonize_label(lb, allowed_targets, others_whitelist)
            if cname is None:
                continue
            if cname in allowed_targets:
                fg_mask |= mk
            elif cname in others_whitelist:
                oth_mask |= mk

    oth_mask &= ~fg_mask

    n_dir = str(parsed_n) if parsed_n is not None else "nX"
    base_dir = overlay_dir / n_dir / out_case
    base_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"__{variant_tag}" if variant_tag else ""

    fg_overlay_path = base_dir / f"{file_stem}{suffix}_step{step_idx:02d}_label.png"
    oth_overlay_path = base_dir / f"{file_stem}{suffix}_step{step_idx:02d}_others.png"

    fg_mask_path = None
    oth_mask_path = None

    if save_npy:
        if mask_dir is None:
            mask_dir = Path(str(overlay_dir).replace("overlays", "masks"))
        npy_base = mask_dir / n_dir / out_case
        npy_base.mkdir(parents=True, exist_ok=True)
        fg_mask_path = npy_base / f"{file_stem}{suffix}_step{step_idx:02d}_label.npy"
        oth_mask_path = npy_base / f"{file_stem}{suffix}_step{step_idx:02d}_others.npy"
        np.save(fg_mask_path, fg_mask.astype(np.uint8))
        np.save(oth_mask_path, oth_mask.astype(np.uint8))

    save_overlay(
        image_pil=image_pil,
        mask_bool=fg_mask,
        out_png_path=fg_overlay_path,
        rgba=fg_rgba,
        title=f"Step {step_idx}: TARGET",
    )
    save_overlay(
        image_pil=image_pil,
        mask_bool=oth_mask,
        out_png_path=oth_overlay_path,
        rgba=oth_rgba,
        title=f"Step {step_idx}: OBJECTS",
    )

    img_area = int(height * width)
    fg_pixels = int(fg_mask.sum())
    oth_pixels = int(oth_mask.sum())

    step_meta = {
        "step_idx": step_idx,
        "prompt": prompt_text,
        "image_size": [height, width],
        "img_area": img_area,
        "fg_pixels": fg_pixels,
        "oth_pixels": oth_pixels,
        "fg_ratio": (fg_pixels / img_area) if img_area > 0 else 0.0,
        "oth_ratio": (oth_pixels / img_area) if img_area > 0 else 0.0,
        "fg_overlay": str(fg_overlay_path),
        "oth_overlay": str(oth_overlay_path),
    }

    if save_npy:
        step_meta.update(
            {
                "fg_mask_npy": str(fg_mask_path),
                "oth_mask_npy": str(oth_mask_path),
            }
        )

    print(f"[step {step_idx:02d}] prompt='{prompt_text}'")
    print(
        f"[step {step_idx:02d}] saved: "
        f"{fg_overlay_path.name}, {oth_overlay_path.name}"
        + (
            ""
            if not save_npy
            else f", {fg_mask_path.name}, {oth_mask_path.name}"
        )
    )

    return step_meta


# ----------------------------------------------------------------------
# Main per-image pipeline
# ----------------------------------------------------------------------
def process_one_image(
    model,
    img_path: Path,
    out_base: Path,
    fail_jsonl: Path,
    tie_jsonl: Path,
    save_npy: bool,
    fg_rgba,
    oth_rgba,
    skip_existing: bool,
) -> None:
    overlay_dir = out_base / "overlays"
    mask_dir = out_base / "masks"
    meta_dir_root = out_base / "meta"

    parsed_label, parsed_n, parsed_case = parse_from_path(img_path)
    out_case = "conflicting"

    if parsed_label is None:
        raise ValueError(f"Failed to parse label from path: {img_path}")

    file_stem = sanitize_name(img_path.stem)

    n_dir = str(parsed_n) if parsed_n is not None else "nX"
    meta_dir = meta_dir_root / n_dir / out_case
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{file_stem}.json"

    if skip_existing and meta_path.exists():
        print(f"[skip] meta exists -> {meta_path}")
        return

    image_pil = Image.open(img_path).convert("RGB")
    target_label = parsed_label
    allowed_targets = {norm(target_label)}

    (
        pred_best,
        class_found,
        class_area,
        sweep_prompts,
        class_unique,
        class_unique_ratio,
        sweep_obj_hits,
        sweep_first_obj_found,
        sweep_first_obj_hits,
    ) = predict_conflict_class(model, image_pil, target_label)

    candidates = [c for c in CLASS_OBJECTS.keys() if c != target_label]
    pri_candidates = [c for c in candidates if sweep_first_obj_found.get(c, False)]
    use_first = bool(pri_candidates)
    pool = pri_candidates if pri_candidates else candidates

    def ranking_key(c: str):
        if use_first:
            return (
                sweep_first_obj_hits[c],
                class_unique[c],
                class_found[c],
                class_area[c],
            )
        return (class_unique[c], class_found[c], class_area[c])

    best_key = None
    ties: list[str] = []
    for cls in pool:
        k = ranking_key(cls)
        if best_key is None or k > best_key:
            best_key = k
            ties = [cls]
        elif k == best_key:
            ties.append(cls)

    if len(ties) > 1:
        append_jsonl(
            tie_jsonl,
            {
                "img_path": str(img_path),
                "label": target_label,
                "case": out_case,
                "ties": ties,
                "keys": {c: ranking_key(c) for c in ties},
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(f"[tie] {len(ties)} classes tied -> logged to {tie_jsonl}")
        selected_classes = ties
    else:
        selected_classes = [max(pool, key=ranking_key)]

    variants_meta = {}
    for cls in selected_classes:
        whitelist_objects = CLASS_OBJECTS[cls]
        prompt = f"{target_label}. " + ". ".join(whitelist_objects) + "."
        parts = [p.strip() for p in prompt.split(".") if p.strip()]
        final_prompt = ". ".join(parts) + "."

        variant_tag = f"fc-{sanitize_name(cls)}" if len(selected_classes) > 1 else None

        step_meta = run_step_and_save(
            model=model,
            image_pil=image_pil,
            prompt_text=final_prompt,
            step_idx=1,
            allowed_targets=allowed_targets,
            others_whitelist={norm(x) for x in whitelist_objects},
            fg_rgba=fg_rgba,
            oth_rgba=oth_rgba,
            overlay_dir=overlay_dir,
            file_stem=file_stem,
            parsed_n=parsed_n,
            out_case=out_case,
            save_npy=save_npy,
            mask_dir=mask_dir,
            variant_tag=variant_tag,
        )

        if whitelist_objects and step_meta.get("oth_pixels", 0) == 0:
            append_jsonl(
                fail_jsonl,
                {
                    "img_path": str(img_path),
                    "label": target_label,
                    "case": out_case,
                    "reason": "no_others_detected",
                    "others_whitelist": [norm(x) for x in whitelist_objects],
                    "prompt": step_meta.get("prompt", ""),
                    "fg_pixels": step_meta.get("fg_pixels", 0),
                    "oth_pixels": step_meta.get("oth_pixels", 0),
                    "variant": variant_tag if variant_tag else "single",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )

        variants_meta[cls] = {
            "prompt": final_prompt,
            "step": step_meta,
        }

    meta_row = {
        "_sweep_note": (
            "priority: first-object presence → unique-object coverage → "
            "total instances → area"
        ),
        "img_path": str(img_path),
        "file_stem": file_stem,
        "label": target_label,
        "n_objects": parsed_n,
        "case": out_case,
        "allowed_targets": sorted(list(allowed_targets)),
        "sweep_prompts": sweep_prompts,
        "sweep_found_instances": class_found,
        "sweep_area": class_area,
        "sweep_unique": class_unique,
        "sweep_unique_ratio": class_unique_ratio,
        "sweep_obj_hits": sweep_obj_hits,
        "sweep_first_obj_found": sweep_first_obj_found,
        "sweep_first_obj_hits": sweep_first_obj_hits,
    }

    if len(selected_classes) > 1:
        meta_row["tie_candidates"] = selected_classes
        meta_row["variants"] = variants_meta
    else:
        cls = selected_classes[0]
        meta_row["foreign_class"] = cls
        meta_row["whitelist_objects"] = CLASS_OBJECTS[cls]
        meta_row["others_whitelist"] = sorted(norm(x) for x in CLASS_OBJECTS[cls])
        meta_row["step"] = variants_meta[cls]["step"]

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_row, f, ensure_ascii=False, indent=2)
    print(f"[meta] saved JSON -> {meta_path}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] JSONL parse error (line {ln}): {e}")
                continue
            yield ln, obj


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangSAM conflicting segmentation runner (JSONL or single image)."
    )
    parser.add_argument(
        "--jsonl",
        help="Input JSONL (each line: {'image': ..., 'label': ..., 'objects': ...}).",
    )
    parser.add_argument(
        "--img",
        help="Single image path. Used when --jsonl is not provided.",
    )
    parser.add_argument(
        "--out-base",
        required=True,
        help="Base output directory.",
    )
    parser.add_argument(
        "--fail-jsonl",
        default=None,
        help="Failure logging JSONL path (default: out-base/failures.jsonl).",
    )
    parser.add_argument(
        "--tie-jsonl",
        default=None,
        help="Tie logging JSONL path (default: out-base/tie.jsonl).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images if meta JSON already exists.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save-npy",
        dest="save_npy",
        action="store_true",
        help="Save .npy masks.",
    )
    group.add_argument(
        "--no-save-npy",
        dest="save_npy",
        action="store_false",
        help="Do not save .npy masks.",
    )
    parser.set_defaults(save_npy=True)

    args = parser.parse_args()

    out_base = Path(args.out_base).resolve()
    fail_jsonl = (
        (out_base / "failures.jsonl").resolve()
        if args.fail_jsonl is None
        else Path(args.fail_jsonl).resolve()
    )
    tie_jsonl = (
        (out_base / "tie.jsonl").resolve()
        if args.tie_jsonl is None
        else Path(args.tie_jsonl).resolve()
    )

    fg_rgba = (0.0, 0.85, 1.0, 0.35)
    oth_rgba = (1.0, 0.1, 0.7, 0.35)

    model = LangSAM()

    if args.jsonl:
        jsonl_path = Path(args.jsonl).resolve()
        for ln, obj in iter_jsonl(jsonl_path):
            img_str = obj.get("image") or obj.get("img")
            if not img_str:
                print(f"[warn] line {ln}: missing 'image' key. Skipping.")
                continue

            img_path = Path(img_str).expanduser().resolve()
            if not img_path.exists():
                print(f"[warn] line {ln}: image not found: {img_path}")
                continue

            try:
                process_one_image(
                    model=model,
                    img_path=img_path,
                    out_base=out_base,
                    fail_jsonl=fail_jsonl,
                    tie_jsonl=tie_jsonl,
                    save_npy=args.save_npy,
                    fg_rgba=fg_rgba,
                    oth_rgba=oth_rgba,
                    skip_existing=args.skip_existing,
                )
            except Exception as e:
                append_jsonl(
                    fail_jsonl,
                    {
                        "img_path": str(img_path),
                        "reason": f"exception: {type(e).__name__}: {e}",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                print(f"[error] processing failed: {img_path} -> {e}")
    else:
        if not args.img:
            raise SystemExit("Missing input: specify either --jsonl or --img.")
        img_path = Path(args.img).resolve()
        process_one_image(
            model=model,
            img_path=img_path,
            out_base=out_base,
            fail_jsonl=fail_jsonl,
            tie_jsonl=tie_jsonl,
            save_npy=args.save_npy,
            fg_rgba=fg_rgba,
            oth_rgba=oth_rgba,
            skip_existing=args.skip_existing,
        )
