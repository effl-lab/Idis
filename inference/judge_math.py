#!/usr/bin/env python3
"""LLM judge for Idis-math outputs (VLMEvalKit MathVerse protocol, vLLM batch).

  stage 1 : extract the final answer     stage 2 : score against the ground truth
  judge   : Qwen/Qwen3.5-27B (default)
  usage   : python judge_math.py --results-files results/<model>/*.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── in-context examples (from VLMEvalKit's MathVerse judge) ─────────────

EXTRACT_PROMPT = """\
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.


1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\\n\\n(-2, 1)'
Extracted Answer: (-2, 1)


2.
Model response: 'at those points.\\n\\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\\n\\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D


3.
Model response: ' at 1 (there\\'s a closed circle at y = 1), the range in interval notation is \\\\((-4, 1]\\\\).\\n\\nFinal values:\\nDomain: \\\\((-3, 3]\\\\)\\nRange: \\\\((-4, 1]\\\\)'
Extracted Answer: Domain: \\((-3, 3]\\)
Range: \\((-4, 1]\\)


4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn\\'t enough information to solve for \\'y\\'.'
Extracted Answer: null


5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\\n\\nd = 17.6 / cos(38°)\\n\\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3


6.
Model response:  have all the coefficients for the quadratic function:\\n\\\\( f(x) = ax^2 + bx + c \\\\)\\n\\\\( f(x) = -1x^2 - 2x + 1 \\\\)\\n\\nTherefore, the equation for the graphed function \\\\( f \\\\) is:\\n\\\\( f(x) = -x^2 - 2x + 1 \\\\)"'
Extracted answer: f(x) = -x^2 - 2x + 1


7.
Model response: '{prediction}'
Extracted Answer: """

SCORE_PROMPT = """\
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent. Examples of consistent pairs: 0.5m and 50cm, 1/3 and 0.333, 90° and π/2, 1690 and 1690cm² (value matches even if unit is omitted), 847/3 and 282.33 (fraction equals decimal).
If they are consistent, Judement is 1; if they are different, Judement is 0.


[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0


[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()
Choices:
A:2
B:2√{{3}}
C:√{{3}}
D:2√{{2}}
[Standard Answer]: C
[Model_answer] : B:2√{{3}}
Judgement: 0


[Question]: Find the area of the surface.
[Standard Answer]: $1690cm^2$
[Model_answer] : 1690
Judgement: 1


[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0


[Question]: Find the volume.
[Standard Answer]: \\frac{{847}}{{3}} mm^3
[Model_answer] : 282.33
Judgement: 1


[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()
Choices:
A:2
B:2√{{3}}
C:√{{3}}
D:2√{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0


    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""


def build_chat_prompt(tokenizer, user_text: str) -> str:
    """Apply chat template with thinking disabled."""
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def batch_generate(llm, prompts: list, max_tokens: int) -> list:
    """Run vLLM batch generation."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def judge_single_file(results_path, eval_suffix, llm, tokenizer, force=False):
    """Judge one results file using batched vLLM inference."""
    results_path = Path(results_path)
    eval_path = results_path.with_suffix(f".eval.{eval_suffix}.jsonl")

    # Load results
    records = []
    with open(results_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print(f"  Empty file, skipping")
        return None

    # Resume or force restart
    done = {}
    if force and eval_path.exists():
        eval_path.unlink()
    if eval_path.exists():
        with open(eval_path, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[str(r["sample_index"])] = r

    todo = [r for r in records if str(r["sample_index"]) not in done]

    if not todo:
        results_eval = list(done.values())
    else:
        # ── Stage 1: Extract answers (batch) ─────────────────────
        # Truncate very long predictions to keep prompt within max_model_len.
        # We keep the END of the prediction (where answer usually appears).
        MAX_PRED_CHARS = 12000  # ~3000 tokens, leaves headroom for template + output
        extract_prompts = []
        for r in todo:
            prediction = r.get("prediction", r.get("output", ""))
            if len(prediction) > MAX_PRED_CHARS:
                prediction = prediction[-MAX_PRED_CHARS:]
            user = EXTRACT_PROMPT.format(prediction=prediction)
            extract_prompts.append(build_chat_prompt(tokenizer, user))

        print(f"  Extracting {len(extract_prompts)} answers...")
        extracts = batch_generate(llm, extract_prompts, max_tokens=128)

        # ── Stage 2: Score (with exact-match prefetch) ───────────
        score_todo = []  # indices that need LLM scoring
        scores = [None] * len(todo)
        log_scores = [None] * len(todo)

        for i, (r, extract) in enumerate(zip(todo, extracts)):
            if extract.strip() == r["answer"].strip():
                scores[i] = True
                log_scores[i] = "Prefetch succeed (exact match)"
            else:
                score_todo.append(i)

        if score_todo:
            score_prompts = []
            for i in score_todo:
                r = todo[i]
                user = SCORE_PROMPT.format(
                    question_for_eval=r.get("question_for_eval", ""),
                    answer=r["answer"],
                    extract=extracts[i],
                )
                score_prompts.append(build_chat_prompt(tokenizer, user))

            print(f"  Scoring {len(score_prompts)} (after exact-match prefetch)...")
            score_outputs = batch_generate(llm, score_prompts, max_tokens=16)

            for idx, out in zip(score_todo, score_outputs):
                m = re.search(r"[01]", out)
                if m:
                    scores[idx] = (m.group() == "1")
                    log_scores[idx] = f"Succeed: {out[:50]}"
                else:
                    scores[idx] = False
                    log_scores[idx] = f"Unexpected: {out[:100]}"

        # ── Write results ────────────────────────────────────────
        results_eval = list(done.values())
        eval_f = open(eval_path, "a", encoding="utf-8")
        for i, r in enumerate(todo):
            sid = str(r["sample_index"])
            result = {
                "sample_index": sid,
                "answer": r["answer"],
                "extract": extracts[i],
                "score": scores[i],
                "log_score": log_scores[i],
                "problem_version": r.get("problem_version", ""),
            }
            results_eval.append(result)
            done[sid] = result
            eval_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        eval_f.flush()
        eval_f.close()

    # Summary
    all_scores = [r["score"] for r in results_eval]
    acc = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
    new = len(todo)
    resumed = len(done) - new
    print(f"  {results_path.name}: {sum(all_scores)}/{len(all_scores)} = {acc:.2f}% "
          f"(new={new}, resumed={resumed})")
    return acc


def main():
    parser = argparse.ArgumentParser(description="vLLM-based LLM judge for MathVerse")
    parser.add_argument("--results-files", type=str, nargs="+", required=True,
                        help="One or more .jsonl inference result files")
    parser.add_argument("--eval-suffix", type=str, default="local")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs to reduce GPU memory footprint.")
    args = parser.parse_args()

    files = [f for f in args.results_files if Path(f).exists()]
    if not files:
        print("No valid result files found.")
        return

    print(f"Judge model: {args.judge_model}")
    print(f"Files to evaluate: {len(files)}")

    # Load vLLM engine
    print("Loading judge model via vLLM...")
    llm = LLM(
        model=args.judge_model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )

    # Tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    print("Model loaded.\n")

    # Build task label from files: e.g. "glm/handwritten_n4_chunk1of2"
    # Parent dir = model_short, filename minus _s{N}.jsonl = task prefix.
    import re
    def derive_task_label(file_paths):
        labels = set()
        for f in file_paths:
            p = Path(f)
            model = p.parent.name
            stem = re.sub(r"_s\d+$", "", p.stem)
            labels.add(f"{model}/{stem}")
        if len(labels) == 1:
            return next(iter(labels))
        return f"{len(labels)} task groups"
    task_label = derive_task_label(files)

    half = len(files) // 2
    notified_half = False

    failures = []
    for i, fpath in enumerate(files):
        print(f"[{i+1}/{len(files)}] {fpath}")
        try:
            judge_single_file(fpath, args.eval_suffix, llm, tokenizer, args.force)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {fpath} -> {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append((fpath, f"{type(e).__name__}: {e}"))

        if not notified_half and (i + 1) >= half > 0:
            notified_half = True
            print(f"[progress] {task_label}: {i+1}/{len(files)} files")

    if failures:
        print(f"\n{len(files)-len(failures)}/{len(files)} files evaluated. Failed:")
        for fp, msg in failures:
            print(f"  {fp} -> {msg}")
    else:
        print(f"\nAll {len(files)} files evaluated.")


if __name__ == "__main__":
    main()
