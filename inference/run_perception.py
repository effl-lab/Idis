"""Idis-perception / Waterbirds inference.

  --dataset : Background_Challenge {image, label[, objects]} | waterbirds {question_id, image, label, background}
  --prompt  : base | strategy                      (inference/prompts.py)
  decoding  : N samples (T=0.7, top-p=0.95) → <stem>_s{i}.jsonl;  --greedy → single output
  output    : input fields + {text, answer, sample_idx}
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inference.models import LOADERS, BUILDERS, select_key  # noqa: E402
from inference.prompts import PROMPTS  # noqa: E402

IMAGE_SIZE = (512, 512)  # every image is resized to 512x512 before the processor (as in the paper runs)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def load_jsonl(path):
    rows = []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON on line {line_num}: {e}")
    return rows


def build_record(dataset: str, line: dict, question: str) -> dict:
    if dataset == "Background_Challenge":
        rec = {"image": line["image"], "text": question, "label": line["label"]}
        if "objects" in line:
            rec["n_objects"] = line["objects"]
        return rec
    # waterbirds
    return {
        "question_id": line["question_id"],
        "image": line["image"],
        "text": question,
        "label": line["label"],
        "background": line["background"],
    }


def eval_model(args):
    key = select_key(args.model_path)
    model, processor = LOADERS[key](args.model_path)

    questions = get_chunk(load_jsonl(args.question_file), args.num_chunks, args.chunk_idx)
    question = PROMPTS[args.dataset][args.prompt]
    num_samples = 1 if args.greedy else args.num_samples

    # one output file per sample index: <stem>_s{i}.jsonl
    base_root, base_ext = os.path.splitext(os.path.expanduser(args.answers_file))
    base_ext = base_ext or ".jsonl"
    ans_files = []
    for i in range(num_samples):
        out_path = f"{base_root}_s{i}{base_ext}"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ans_files.append(open(out_path, "w", encoding="utf-8"))

    for line in tqdm(questions):
        image = Image.open(os.path.join(args.image_folder, line["image"])).convert("RGB").resize(IMAGE_SIZE)
        payload = BUILDERS[key](processor, image, question)
        inputs = processor(**payload, padding=True, return_tensors="pt").to("cuda")

        gen_kwargs = {"max_new_tokens": args.max_token}
        if args.greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=num_samples,
            )
        generated_ids = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        output_texts = processor.batch_decode(
            [seq[input_len:] for seq in generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        base_record = build_record(args.dataset, line, question)
        for i in range(min(len(output_texts), num_samples)):
            record = dict(base_record)
            record["answer"] = output_texts[i]
            record["sample_idx"] = i
            ans_files[i].write(json.dumps(record, ensure_ascii=False) + "\n")

    for f in ans_files:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(PROMPTS))
    parser.add_argument("--prompt", default="base", choices=["base", "strategy"])
    parser.add_argument("--model-path", required=True,
                        help="Qwen/Qwen3-VL-8B-Thinking | zai-org/GLM-4.1V-9B-Thinking | internlm/Intern-S1-mini | Fancy-MLLM/R1-Onevision-7B-RL")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--question-file", required=True, help="input JSONL (see module docstring)")
    parser.add_argument("--answers-file", default="./answer.jsonl", help="output stem; _s{i}.jsonl is appended")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="do_sample=False, single output")
    parser.add_argument("--max-token", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)
    print(args)
    eval_model(args)
