"""Textual distractors for Idis-math (Sonnet 4.5 via the `claude` CLI).

  input  : MathVerse testmini (Text Dominant / Text Lite / Vision Intensive / Vision Dominant)
  output : text_distractor/n{K}/meta.jsonl with augmented questions
  usage  : python gen_textual_distractors.py --num-distractors K [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

# ── paths ──────────────────────────────────────────────────────────
# Set in main() from --mathverse-root.
IMG_ROOT = OUT_BASE = None

TARGET_PV = {"vision intensive", "vision dominant", "text lite", "text dominant"}

# ── prompt template ────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a math education expert. Your task is to insert distractor sentences \
into a math question to test whether a solver can ignore irrelevant information \
and focus on what actually matters.

I have a math question (with an accompanying figure). I want you to add \
{num_distractors} distractor sentences into the question text. The distractors \
should be inserted between the setup/given conditions and the final question, \
blending naturally into the text.

<original_question>
{question}
</original_question>

<correct_answer>
{answer}
</correct_answer>

<subject>
{subject} / {subfield}
</subject>

Rules:
1. Distractors MUST be mathematically related to the question's domain \
(geometry, algebra, etc.) but MUST NOT affect the correct answer.
2. Distractors should introduce plausible-sounding information (extra angles, \
lengths, relationships, theorems, or observations) that seems relevant but is \
actually irrelevant to solving the problem.
3. The distractors should reward careful thinking about what information is \
actually needed.
4. The correct answer MUST remain exactly {answer} after inserting distractors.
5. Distractors should blend naturally into the question text — they should read \
as if they were always part of the problem statement.
6. Do NOT change, remove, or rephrase the original sentences. Only INSERT new \
distractor sentences.
7. After generating, verify that the answer is still {answer} by mentally solving \
the augmented question. If it's not, regenerate.
8. Output ONLY the complete augmented question (with distractors inserted). \
Do not output the answer, explanation, tags, or anything else.
9. Keep the Choices section exactly as-is at the end."""


def call_claude_cli(prompt: str, image_path: str | None = None,
                    max_retries: int = 3) -> str | None:
    """Call claude CLI in non-interactive mode."""
    for attempt in range(max_retries):
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        if image_path and os.path.isfile(image_path):
            cmd.extend(["--files", image_path])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"  attempt {attempt+1} failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  attempt {attempt+1} timed out")
        except Exception as e:
            print(f"  attempt {attempt+1} error: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** (attempt + 1))

    return None


def load_done(meta_path: str) -> set:
    """Load already-processed sample indices for resume support."""
    done = set()
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done.add(rec["sample_index"])
    return done


def main():
    global IMG_ROOT, OUT_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathverse-root", default=os.environ.get("MATHVERSE_ROOT"),
                    help="directory with testmini.json and testmini/images/ (env: MATHVERSE_ROOT)")
    ap.add_argument("--selected", default=None, help="question file (default: <mathverse-root>/testmini.json)")
    ap.add_argument("--out-base", default=None,
                    help="output root (default: <mathverse-root>/mathverse_aug/text_distractor)")
    ap.add_argument("--out", default=None,
                    help="output path (default: auto from num-distractors)")
    ap.add_argument("--num-distractors", type=int, default=5)
    ap.add_argument("--no-image", action="store_true",
                    help="skip sending image (text-only mode)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print first prompt and exit")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N items (0=all)")
    args = ap.parse_args()
    if not args.mathverse_root:
        ap.error("--mathverse-root (or MATHVERSE_ROOT) is required")
    IMG_ROOT = args.mathverse_root
    OUT_BASE = args.out_base or os.path.join(args.mathverse_root, "mathverse_aug", "text_distractor")
    args.selected = args.selected or os.path.join(args.mathverse_root, "testmini.json")

    # load & filter data
    with open(args.selected, "r") as f:
        raw = json.load(f) if args.selected.endswith(".json") else [json.loads(l) for l in f if l.strip()]
    items = [
        item for item in raw
        if item.get("problem_version", "").strip().lower() in TARGET_PV
        and item.get("question", "").strip()
    ]
    print(f"loaded {len(items)} target items (VI+VD+TL+TD) from {len(raw)} total")

    # resolve output path: text_distractor/n{K}/meta.jsonl
    out_dir = os.path.join(OUT_BASE, f"n{args.num_distractors}")
    out_path = args.out or os.path.join(out_dir, "meta.jsonl")
    out_dir = os.path.dirname(out_path)

    if args.dry_run:
        item = items[0]
        meta = item.get("metadata", {})
        prompt = PROMPT_TEMPLATE.format(
            num_distractors=args.num_distractors,
            question=item["question"],
            answer=item["answer"],
            subject=meta.get("subject", "Math"),
            subfield=meta.get("subfield", ""),
        )
        img_path = os.path.join(IMG_ROOT, item["image"])
        print("=== PROMPT ===")
        print(prompt)
        print(f"\n=== IMAGE === {img_path}")
        print(f"=== CMD === claude -p '...' --output-format text --files {img_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # resume support
    done = load_done(out_path)
    remaining = [it for it in items if it["sample_index"] not in done]
    if done:
        print(f"resuming: {len(done)} done, {len(remaining)} remaining")

    if args.limit:
        remaining = remaining[:args.limit]

    print(f"processing {len(remaining)} items...")

    print(f"output: {out_path}")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, item in enumerate(remaining):
            sid = item["sample_index"]
            meta = item.get("metadata", {})
            print(f"[{i+1}/{len(remaining)}] sample_index={sid} "
                  f"pv={item['problem_version']}")

            prompt = PROMPT_TEMPLATE.format(
                num_distractors=args.num_distractors,
                question=item["question"],
                answer=item["answer"],
                subject=meta.get("subject", "Math"),
                subfield=meta.get("subfield", ""),
            )

            img_path = None
            if not args.no_image:
                img_path = os.path.join(IMG_ROOT, item["image"])

            augmented_question = call_claude_cli(prompt, img_path)

            if augmented_question is None:
                print(f"  FAILED, skipping")
                continue

            record = {
                "sample_index": sid,
                "problem_version": item["problem_version"],
                "original_question": item["question"],
                "augmented_question": augmented_question,
                "answer": item["answer"],
                "image": item["image"],
                "num_distractors": args.num_distractors,
                "metadata": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                print(f"  progress: {i+1}/{len(remaining)}")

    final_done = load_done(out_path)
    print(f"done: {len(final_done)} total records in {out_path}")


if __name__ == "__main__":
    main()
