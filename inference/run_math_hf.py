#!/usr/bin/env python3
"""Idis-math inference with HuggingFace generate (see run_math_vllm.py for the vLLM version).

  variants : original | aligned | conflicting | irrelevant | handwritten | mathwriting | text_distractor
  output   : {output_dir}/{variant}_n{N}_s{k}.jsonl, k = 0..K-1
  usage    : python run_math_hf.py --model-path <hf id> --tasks "aligned:1,conflicting:4" \
                 --num-samples 5 --output-dir <dir>
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── model registry (inference/models.py) + compatibility shims ──────────────
import transformers
if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = None

import huggingface_hub.dataclasses as _hf_dc
_orig_type_validator = _hf_dc.type_validator
def _patched_type_validator(name, value, expected_type):
    if value is None and expected_type is bool:
        return
    return _orig_type_validator(name, value, expected_type)
_hf_dc.type_validator = _patched_type_validator

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from inference.models import LOADERS, BUILDERS, select_key

# ── paths ────────────────────────────────────────────────────────────────
# Set from --mathverse-root / --aug-root / --library-dir at startup.
TESTMINI = IMG_ROOT = BASE_DIR = None
LIBRARY_DIR = "library"

# ── VLMEvalKit-aligned prompt suffix ─────────────────────────────────────
SUFFIX = (
    "\nYou first think through the reasoning process as an internal monologue, "
    "enclosed within <think> </think> tags. "
    "Then, provide your final answer enclosed within \\boxed{}."
)

# ── Model-specific max_new_tokens ────────────────────────────────────────
MAX_NEW_TOKENS = {
    "qwen3-vl-8b-thinking": 8192,
    "glm":                  8192,
    "intern-s1":            8192,
    "r1-onevision":         8192,
}
DEFAULT_MAX_NEW_TOKENS = 8192


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_testmini_index():
    """Load testmini.json and build sample_index -> entry mapping."""
    with open(TESTMINI, "r", encoding="utf-8") as f:
        full = json.load(f)
    return {str(e["sample_index"]): e for e in full}


# ── dataset loaders ──────────────────────────────────────────────────────

def load_original(testmini_idx):
    """Original MathVerse testmini — all 3,940 samples (all 5 problem_versions)."""
    samples = []
    for sid, entry in testmini_idx.items():
        image = entry.get("image", "")
        if not image:
            continue
        samples.append({
            "sample_index": sid,
            "image_path": os.path.join(IMG_ROOT, image),
            "question": entry.get("question", ""),
            "answer": entry["answer"],
            "question_for_eval": entry.get("question_for_eval", ""),
            "problem_version": entry.get("problem_version", ""),
        })
    return samples


def load_image_augmented(testmini_idx, variant, n_distractor):
    """
    For aligned, handwritten, irrelevant, mathwriting.
    Meta in library/{variant}/meta.jsonl, images referenced by augmented.n{N}.
    Question text comes from testmini.json.
    """
    meta_path = os.path.join(BASE_DIR, LIBRARY_DIR, variant, "meta.jsonl")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f if line.strip()]

    n_key = f"n{n_distractor}"
    samples = []
    for entry in meta:
        sid = str(entry["sample_index"])
        sel = testmini_idx.get(sid)
        if sel is None:
            continue

        aug_map = entry.get("augmented", {})
        if n_key not in aug_map:
            continue

        image_path = os.path.join(BASE_DIR, aug_map[n_key])
        if not os.path.exists(image_path):
            continue

        # Question from testmini.json
        question = sel.get("question", "")

        samples.append({
            "sample_index": sid,
            "image_path": image_path,
            "question": question,
            "answer": sel["answer"],
            "question_for_eval": sel.get("question_for_eval", ""),
            "problem_version": entry.get("problem_version", sel.get("problem_version", "")),
        })
    return samples


def load_text_distractor(testmini_idx, n_distractor):
    """
    Text distractor: augmented question + original image.
    Meta in text_distractor/n{N}/meta.jsonl.
    """
    meta_path = os.path.join(BASE_DIR, "text_distractor", f"n{n_distractor}", "meta.jsonl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"text_distractor meta not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f if line.strip()]

    samples = []
    for entry in meta:
        sid = str(entry["sample_index"])
        image_path = os.path.join(IMG_ROOT, entry["image"])
        if not os.path.exists(image_path):
            continue

        samples.append({
            "sample_index": sid,
            "image_path": image_path,
            "question": entry["augmented_question"],
            "answer": entry["answer"],
            "question_for_eval": entry.get("original_question", entry["augmented_question"]),
            "problem_version": entry.get("problem_version", ""),
        })
    return samples


def load_conflicting(testmini_idx, n_distractor):
    """
    Conflicting: Multiple conflict_labels per sample_index.
    Uses compound key '{sid}__{conflict_label}' to make each entry unique
    so resume logic works correctly.
    """
    meta_path = os.path.join(BASE_DIR, LIBRARY_DIR, "conflicting", "meta.jsonl")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f if line.strip()]

    n_key = f"n{n_distractor}"
    samples = []
    for entry in meta:
        sid = str(entry["sample_index"])
        sel = testmini_idx.get(sid)
        if sel is None:
            continue

        aug_map = entry.get("augmented", {})
        if n_key not in aug_map:
            continue

        image_path = os.path.join(BASE_DIR, aug_map[n_key])
        if not os.path.exists(image_path):
            continue

        conflict_label = entry.get("conflict_label", "")
        compound_key = f"{sid}__{conflict_label}"

        samples.append({
            "sample_index": compound_key,
            "original_sample_index": sid,
            "conflict_label": conflict_label,
            "image_path": image_path,
            "question": sel.get("question", ""),
            "answer": sel["answer"],
            "question_for_eval": sel.get("question_for_eval", ""),
            "problem_version": entry.get("problem_version", sel.get("problem_version", "")),
        })
    return samples


def load_dataset(variant, n_distractor, testmini_idx):
    if variant == "original":
        return load_original(testmini_idx)
    elif variant == "text_distractor":
        return load_text_distractor(testmini_idx, n_distractor)
    elif variant == "conflicting":
        return load_conflicting(testmini_idx, n_distractor)
    elif variant in ("aligned", "handwritten", "irrelevant", "mathwriting"):
        return load_image_augmented(testmini_idx, variant, n_distractor)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ── inference ────────────────────────────────────────────────────────────

def parse_tasks(tasks_str):
    """Parse task string like 'original:1,aligned:1,aligned:2' into list of (variant, n)."""
    tasks = []
    for item in tasks_str.split(","):
        item = item.strip()
        if ":" in item:
            variant, n = item.split(":")
            tasks.append((variant, int(n)))
        else:
            tasks.append((item, 1))
    return tasks


def run_single_task(variant, n_distractor, model, processor, key,
                    testmini_idx, args, task_idx=0, total_tasks=1,
                    model_short=""):
    """Run inference for one (variant, n_distractor) combination."""

    samples = load_dataset(variant, n_distractor, testmini_idx)
    task_label = f"[{model_short}] {variant}_n{n_distractor}"
    print(f"\n{'='*60}")
    print(f"Task {task_idx+1}/{total_tasks}: {variant} n={n_distractor}, total: {len(samples)}")

    # Chunking: split samples across num_chunks, take chunk_idx
    if args.num_chunks > 1:
        import math
        chunk_size = math.ceil(len(samples) / args.num_chunks)
        s = args.chunk_idx * chunk_size
        e = min(s + chunk_size, len(samples))
        samples = samples[s:e]
        print(f"Chunk {args.chunk_idx}/{args.num_chunks}: samples [{s}:{e}] = {len(samples)}")

    if not samples:
        print("No samples found. Skipping.")
        return

    # max_new_tokens
    max_new_tokens = args.max_token or MAX_NEW_TOKENS.get(key, DEFAULT_MAX_NEW_TOKENS)

    # Output file naming (include chunk suffix if applicable)
    if variant == "original":
        base_name = "original"
    else:
        base_name = f"{variant}_n{n_distractor}"
    if args.num_chunks > 1:
        base_name += f"_chunk{args.chunk_idx}of{args.num_chunks}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Open K output files + load resume state
    ans_files = []
    done_sets = []
    for s in range(args.num_samples):
        fpath = os.path.join(args.output_dir, f"{base_name}_s{s}.jsonl")
        done = set()
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["sample_index"])
        done_sets.append(done)
        ans_files.append(open(fpath, "a", encoding="utf-8"))

    # Find samples that need at least one more generation
    all_done = set.intersection(*done_sets) if done_sets else set()
    remaining = [s for s in samples if s["sample_index"] not in all_done]
    if len(remaining) < len(samples):
        print(f"Resuming: {len(samples) - len(remaining)} fully done, {len(remaining)} remaining")

    print(f"max_new_tokens: {max_new_tokens}, "
          f"samples: {args.num_samples}, temp: {args.temperature}")

    half = len(remaining) // 2
    notified_half = False
    total = 0

    for sample in tqdm(remaining, desc=f"{variant}_n{n_distractor}"):
        sid = sample["sample_index"]
        question = sample["question"] + SUFFIX

        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {sample['image_path']}: {e}")
            continue

        # Determine which sample slots still need generation
        needed = [s for s in range(args.num_samples) if sid not in done_sets[s]]
        if not needed:
            continue

        payload = BUILDERS[key](processor, image, question)
        inputs = processor(**payload, padding=True, return_tensors="pt").to("cuda")
        input_len = inputs.input_ids.shape[1]

        # Generate all missing samples in one batched call
        # (prefill done once, decode parallel across N sequences)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=len(needed),
        )
        # Each returned sequence has full ids; trim prefix (input) per sequence
        generated_ids_trimmed = [out_ids[input_len:] for out_ids in generated_ids]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for i, s in enumerate(needed):
            record = {
                "sample_index": sid,
                "image": sample["image_path"],
                "question": question,
                "answer": sample["answer"],
                "question_for_eval": sample.get("question_for_eval", ""),
                "prediction": output_texts[i],
                "problem_version": sample["problem_version"],
                "sample_id": s,
            }
            ans_files[s].write(json.dumps(record, ensure_ascii=False) + "\n")
            ans_files[s].flush()

        total += 1

        if not notified_half and total >= half > 0:
            notified_half = True
            print(f"[progress] {task_label}: {total}/{len(remaining)}")

    for f in ans_files:
        f.close()

    print(f"Done: {total} samples x {args.num_samples} = "
          f"{args.output_dir}/{base_name}_s{{0..{args.num_samples-1}}}.jsonl")


def run_inference(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    seed_everything(args.seed)

    # Load model once
    key = select_key(args.model_path)
    print(f"Model: {args.model_path} (key={key})")
    model, processor = LOADERS[key](args.model_path)

    # Load testmini index once
    testmini_idx = load_testmini_index()

    # Build task list
    if args.tasks:
        task_list = parse_tasks(args.tasks)
    else:
        task_list = [(args.variant, args.n_distractor)]

    print(f"Tasks: {len(task_list)} — {task_list}")
    print(f"Sampling: {args.num_samples} samples, "
          f"temperature={args.temperature}, top_p={args.top_p}")

    # Derive model short name for notifications
    model_short = os.path.basename(args.model_path)

    # Run each task sequentially (model stays loaded, GPU held)
    for i, (variant, n_distractor) in enumerate(task_list):
        run_single_task(
            variant, n_distractor,
            model, processor, key,
            testmini_idx, args,
            task_idx=i, total_tasks=len(task_list),
            model_short=model_short,
        )

    print(f"\n{'='*60}")
    print(f"All {len(task_list)} tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MathVerse Sampling Experiment")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated tasks: 'original:1,aligned:1,aligned:2,...' "
                             "(overrides --variant/--n-distractor)")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["original", "aligned", "handwritten", "irrelevant",
                                 "mathwriting", "text_distractor", "conflicting"])
    parser.add_argument("--n-distractor", type=int, default=1)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-token", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Split each task's samples into N chunks")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Which chunk to process (0-indexed)")
    parser.add_argument("--mathverse-root", type=str, default=os.environ.get("MATHVERSE_ROOT"),
                        help="directory with testmini.json and testmini/images/ (env: MATHVERSE_ROOT)")
    parser.add_argument("--aug-root", type=str, default=None,
                        help="Idis-math root with library/ and text_distractor/ (default: <mathverse-root>/mathverse_aug)")
    parser.add_argument("--library-dir", type=str, default="library",
                        help="subdirectory of aug-root holding aligned/conflicting/irrelevant/...")
    args = parser.parse_args()
    if not args.tasks and not args.variant:
        parser.error("Either --tasks or --variant is required")
    if not args.mathverse_root:
        parser.error("--mathverse-root (or MATHVERSE_ROOT) is required")
    TESTMINI = os.path.join(args.mathverse_root, "testmini.json")
    IMG_ROOT = os.path.join(args.mathverse_root, "testmini", "images")
    BASE_DIR = args.aug_root or os.path.join(args.mathverse_root, "mathverse_aug")
    LIBRARY_DIR = args.library_dir
    print(args)
    run_inference(args)
