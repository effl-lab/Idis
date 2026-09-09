#!/usr/bin/env python3
"""Idis-math inference with vLLM (used for the paper runs).

  variants : original | aligned | conflicting | irrelevant | handwritten | mathwriting | text_distractor
  output   : {output_dir}/{variant}_n{N}_s{k}.jsonl, k = 0..K-1  (resumable)
  usage    : python run_math_vllm.py --model-path <hf id> --tasks "aligned:1,conflicting:4" \
                 --num-samples 5 --output-dir <dir>
"""

import argparse
import json
import math
import os
import sys

from PIL import Image
from tqdm import tqdm

from vllm import LLM, SamplingParams

# ── paths ────────────────────────────────────────────────────────────────
# Set from --mathverse-root / --aug-root at startup.
TESTMINI = IMG_ROOT = BASE_DIR = None

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

# ── Model key detection ─────────────────────────────────────────────────
MODEL_KEYS = ["qwen3-vl-8b-thinking", "glm", "intern-s1", "r1-onevision"]

# Models that need enable_thinking in chat template
THINKING_MODELS = {"intern-s1"}

# Models that need trust_remote_code
TRUST_REMOTE_CODE_MODELS = {"intern-s1"}


def select_key(path: str) -> str:
    path_lower = path.lower()
    for k in MODEL_KEYS:
        if k in path_lower:
            return k
    raise ValueError(f"unsupported model: {path}")


# ── dataset loaders ─────────────────────────────────────────────────────

def load_testmini_index():
    with open(TESTMINI, "r", encoding="utf-8") as f:
        full = json.load(f)
    return {str(e["sample_index"]): e for e in full}


def load_original(testmini_idx):
    samples = []
    for sid, entry in testmini_idx.items():
        image = entry.get("image", "")
        if not image:
            continue
        samples.append({
            "sample_index": sid,
            "image_path": os.path.join(IMG_ROOT, image),
            "question": entry.get("question", ""),
            "query_wo": entry.get("query_wo", entry.get("question", "")),
            "query_cot": entry.get("query_cot", entry.get("question", "")),
            "answer": entry["answer"],
            "question_for_eval": entry.get("question_for_eval", ""),
            "problem_version": entry.get("problem_version", ""),
        })
    return samples


def load_image_augmented(testmini_idx, variant, n_distractor, library_dir="library"):
    meta_path = os.path.join(BASE_DIR, library_dir, variant, "meta.jsonl")
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
        samples.append({
            "sample_index": sid,
            "image_path": image_path,
            "question": sel.get("question", ""),
            "query_wo": sel.get("query_wo", sel.get("question", "")),
            "query_cot": sel.get("query_cot", sel.get("question", "")),
            "answer": sel["answer"],
            "question_for_eval": sel.get("question_for_eval", ""),
            "problem_version": entry.get("problem_version", sel.get("problem_version", "")),
        })
    return samples


def load_text_distractor(testmini_idx, n_distractor):
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


def load_conflicting(testmini_idx, n_distractor, library_dir="library"):
    meta_path = os.path.join(BASE_DIR, library_dir, "conflicting", "meta.jsonl")
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
            "query_wo": sel.get("query_wo", sel.get("question", "")),
            "query_cot": sel.get("query_cot", sel.get("question", "")),
            "answer": sel["answer"],
            "question_for_eval": sel.get("question_for_eval", ""),
            "problem_version": entry.get("problem_version", sel.get("problem_version", "")),
        })
    return samples


def load_dataset(variant, n_distractor, testmini_idx, library_dir="library"):
    if variant == "original":
        return load_original(testmini_idx)
    elif variant == "text_distractor":
        return load_text_distractor(testmini_idx, n_distractor)
    elif variant == "conflicting":
        return load_conflicting(testmini_idx, n_distractor, library_dir=library_dir)
    elif variant in ("aligned", "handwritten", "irrelevant", "mathwriting"):
        return load_image_augmented(testmini_idx, variant, n_distractor, library_dir=library_dir)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ── prompt building ─────────────────────────────────────────────────────

def build_chat_prompt(processor, image, question, key, force_no_thinking: bool = False):
    """Build chat prompt using the processor's chat template."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    kwargs = {}
    if key in THINKING_MODELS and not force_no_thinking:
        kwargs["enable_thinking"] = True
    elif key in THINKING_MODELS and force_no_thinking:
        kwargs["enable_thinking"] = False

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    return prompt


# ── inference ────────────────────────────────────────────────────────────

def parse_tasks(tasks_str):
    tasks = []
    for item in tasks_str.split(","):
        item = item.strip()
        if ":" in item:
            variant, n = item.split(":")
            tasks.append((variant, int(n)))
        else:
            tasks.append((item, 1))
    return tasks


def run_single_task(variant, n_distractor, llm, processor, key,
                    testmini_idx, args, task_idx=0, total_tasks=1,
                    model_short=""):
    """Run batch inference for one (variant, n_distractor) combination."""
    samples = load_dataset(variant, n_distractor, testmini_idx,
                           library_dir=args.library_dir)
    task_label = f"[{model_short}] {variant}_n{n_distractor}"
    print(f"\n{'='*60}")
    print(f"Task {task_idx+1}/{total_tasks}: {variant} n={n_distractor}, total: {len(samples)}")

    # Chunking
    if args.num_chunks > 1:
        chunk_size = math.ceil(len(samples) / args.num_chunks)
        s = args.chunk_idx * chunk_size
        e = min(s + chunk_size, len(samples))
        samples = samples[s:e]
        print(f"Chunk {args.chunk_idx}/{args.num_chunks}: samples [{s}:{e}] = {len(samples)}")

    if not samples:
        print("No samples found. Skipping.")
        return

    max_new_tokens = args.max_token or MAX_NEW_TOKENS.get(key, DEFAULT_MAX_NEW_TOKENS)

    # Output file naming
    if variant == "original":
        base_name = "original"
    else:
        base_name = f"{variant}_n{n_distractor}"
    if args.num_chunks > 1:
        base_name += f"_chunk{args.chunk_idx}of{args.num_chunks}"
    if args.output_prefix:
        base_name = f"{args.output_prefix}{base_name}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load resume state
    done_sets = []
    for s_idx in range(args.num_samples):
        fpath = os.path.join(args.output_dir, f"{base_name}_s{s_idx + args.sample_offset}.jsonl")
        done = set()
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["sample_index"])
        done_sets.append(done)

    all_done = set.intersection(*done_sets) if done_sets else set()
    remaining = [s for s in samples if s["sample_index"] not in all_done]
    if len(remaining) < len(samples):
        print(f"Resuming: {len(samples) - len(remaining)} fully done, {len(remaining)} remaining")

    if not remaining:
        print("All samples already done. Skipping.")
        return

    print(f"max_new_tokens: {max_new_tokens}, "
          f"samples: {args.num_samples}, temp: {args.temperature}")

    # Build all prompts and load images
    print("Building prompts and loading images...")
    batch_inputs = []
    valid_samples = []

    for sample in tqdm(remaining, desc="preparing"):
        # prompt-field controls which testmini field to use; SUFFIX only for default "question".
        if args.prompt_field == "question":
            question = sample["question"] + SUFFIX
        else:
            question = sample.get(args.prompt_field, sample["question"])
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {sample['image_path']}: {e}")
            continue

        prompt = build_chat_prompt(processor, image, question, key,
                                   force_no_thinking=args.force_no_thinking)
        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        })
        valid_samples.append(sample)

    print(f"Prepared {len(valid_samples)} inputs for batch inference")

    # Filter out inputs whose prompt would exceed max_model_len (leave headroom for generation)
    # Tokenize prompts to check length
    max_prompt_len = args.max_model_len - max_new_tokens
    filtered_inputs = []
    filtered_samples = []
    skipped = 0
    for inp, sample in zip(batch_inputs, valid_samples):
        # Use processor's tokenizer to estimate length
        try:
            token_ids = processor.tokenizer(inp["prompt"], return_tensors=None, add_special_tokens=False)["input_ids"]
            prompt_len = len(token_ids)
            # Also account for image tokens (rough estimate: use hf image processor)
        except Exception:
            prompt_len = 0
        # Simple check: reject if text prompt alone is too long (image adds more)
        # We'll let vLLM handle the actual check, but pre-filter obvious cases
        if prompt_len > max_prompt_len:
            skipped += 1
            continue
        filtered_inputs.append(inp)
        filtered_samples.append(sample)

    if skipped > 0:
        print(f"Pre-filtered {skipped} samples with prompt too long")
    batch_inputs = filtered_inputs
    valid_samples = filtered_samples

    # vLLM sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_new_tokens,
        n=args.num_samples,
    )

    # Run batch inference (vLLM will skip requests that fail validation)
    print(f"Running vLLM batch inference on {len(batch_inputs)} samples...")
    try:
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    except ValueError as e:
        # If any single prompt fails, fall back to per-sample try/except
        print(f"Batch failed: {e}")
        print("Falling back to per-sample inference...")
        outputs = []
        for inp in tqdm(batch_inputs, desc="per-sample fallback"):
            try:
                out = llm.generate([inp], sampling_params=sampling_params)
                outputs.extend(out)
            except Exception as e:
                print(f"  [SKIP sample] {e}")
                outputs.append(None)

    # Open output files for writing
    ans_files = []
    for s_idx in range(args.num_samples):
        fpath = os.path.join(args.output_dir, f"{base_name}_s{s_idx + args.sample_offset}.jsonl")
        ans_files.append(open(fpath, "a", encoding="utf-8"))

    # Write results
    for sample, output in zip(valid_samples, outputs):
        if output is None:
            continue
        sid = sample["sample_index"]
        # prompt-field controls which testmini field to use; SUFFIX only for default "question".
        if args.prompt_field == "question":
            question = sample["question"] + SUFFIX
        else:
            question = sample.get(args.prompt_field, sample["question"])

        for s_idx, completion in enumerate(output.outputs):
            if sid in done_sets[s_idx]:
                continue

            record = {
                "sample_index": sid,
                "image": sample["image_path"],
                "question": question,
                "answer": sample["answer"],
                "question_for_eval": sample.get("question_for_eval", ""),
                "prediction": completion.text,
                "problem_version": sample["problem_version"],
                "sample_id": s_idx,
            }
            ans_files[s_idx].write(json.dumps(record, ensure_ascii=False) + "\n")
            ans_files[s_idx].flush()

    for f in ans_files:
        f.close()

    print(f"done: {task_label} ({len(valid_samples)} samples x {args.num_samples})")

    s_lo = args.sample_offset
    s_hi = args.sample_offset + args.num_samples - 1
    print(f"Done: {len(valid_samples)} samples x {args.num_samples} = "
          f"{args.output_dir}/{base_name}_s{{{s_lo}..{s_hi}}}.jsonl")


def run_inference(args):
    key = select_key(args.model_path)
    print(f"Model: {args.model_path} (key={key})")

    # Load vLLM engine
    trust_remote_code = key in TRUST_REMOTE_CODE_MODELS
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=trust_remote_code,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 1},
    )

    # Load processor for chat template
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=trust_remote_code
    )

    # Load testmini index
    testmini_idx = load_testmini_index()

    # Build task list
    if args.tasks:
        task_list = parse_tasks(args.tasks)
    else:
        task_list = [(args.variant, args.n_distractor)]

    print(f"Tasks: {len(task_list)} — {task_list}")
    print(f"Sampling: {args.num_samples} samples, "
          f"temperature={args.temperature}, top_p={args.top_p}")

    model_short = os.path.basename(args.model_path)

    for i, (variant, n_distractor) in enumerate(task_list):
        run_single_task(
            variant, n_distractor,
            llm, processor, key,
            testmini_idx, args,
            task_idx=i, total_tasks=len(task_list),
            model_short=model_short,
        )

    print(f"\n{'='*60}")
    print(f"All {len(task_list)} tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MathVerse vLLM Batch Inference")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated tasks: 'original:1,aligned:1,...'")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["original", "aligned", "handwritten", "irrelevant",
                                 "mathwriting", "text_distractor", "conflicting"])
    parser.add_argument("--n-distractor", type=int, default=1)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-offset", type=int, default=0,
                        help="Shift output file index by this amount (e.g. offset=1 + num_samples=4 => s1..s4)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-token", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # vLLM specific
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Max sequence length for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--library-dir", type=str, default="library",
                        help="subdirectory of aug-root holding aligned/conflicting/irrelevant/...")
    parser.add_argument("--prompt-field", type=str, default="question",
                        choices=["question", "query_wo", "query_cot"],
                        help="Which testmini field to use as the question. "
                             "'question' appends VLMEvalKit thinking SUFFIX; "
                             "'query_wo' / 'query_cot' use the testmini-provided wording verbatim.")
    parser.add_argument("--force-no-thinking", action="store_true",
                        help="For intern-s1: pass enable_thinking=False to chat template "
                             "(overrides THINKING_MODELS default).")
    parser.add_argument("--output-prefix", type=str, default="",
                        help="Prefix prepended to output filename, e.g. 'query_cot-' "
                             "produces 'query_cot-conflicting_n4_s0.jsonl'.")
    parser.add_argument("--mathverse-root", type=str, default=os.environ.get("MATHVERSE_ROOT"),
                        help="directory with testmini.json and testmini/images/ (env: MATHVERSE_ROOT)")
    parser.add_argument("--aug-root", type=str, default=None,
                        help="Idis-math root with library/ and text_distractor/ (default: <mathverse-root>/mathverse_aug)")
    args = parser.parse_args()
    if not args.tasks and not args.variant:
        parser.error("Either --tasks or --variant is required")
    if not args.mathverse_root:
        parser.error("--mathverse-root (or MATHVERSE_ROOT) is required")
    TESTMINI = os.path.join(args.mathverse_root, "testmini.json")
    IMG_ROOT = os.path.join(args.mathverse_root, "testmini", "images")
    BASE_DIR = args.aug_root or os.path.join(args.mathverse_root, "mathverse_aug")
    print(args)
    run_inference(args)
