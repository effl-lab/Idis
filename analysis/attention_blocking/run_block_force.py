"""Attention blocking in the force-answer setting (Fig. 7a).

  arm    ∈ {baseline, block_distractor, block_target}
  layer  ∈ {all, mid_late}            # mid_late = [0.5L, 0.8L)
  query  : tokens generated after <answer>
  block  : attention_logits[q, k] = -inf for k in the phrase tokens
  output : JSONL, one record per (image, condition)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# qwen_vl_utils.process_vision_info — same image preprocessing as inference/models.py
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

# Idis-perception task prompt, byte-level identical to the one used for the teacher-forced
# traces (note the space after "Fish.").
BG_QUESTION = (
    "[Question] Which category best describes the main object in the image? "
    "Choose exactly one from: Dog, Bird, Vehicle, Reptile, Carnivore, Insect, "
    "Instrument, Primate, Fish. \n"
    "Use a thinking process to analyze the problem step-by-step.\n"
    "At the end, provide your answer and clearly indicate it using "
    "<answer>X</answer> format."
)


# ── attention blocker ──────────────────────────────────────────────────────
class AttentionBlocker:
    """Inject an additive 4-D mask into each decoder layer's ``attention_mask`` via forward pre-hooks.

    ``set_block(mask4d, active_layers)`` arms blocking for the following forward passes;
    ``clear()`` restores unblocked (baseline) attention.
    """

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self._mask: torch.Tensor | None = None  # (1, 1, q_len, kv_len) additive
        self._active_layers: set[int] | None = None
        # locate decoder layers — traverse common nested paths.
        # Qwen3-VL: model.model.language_model.layers
        # Qwen2-VL: model.model.layers
        # others: try the candidate paths in order.
        candidates = [
            ("language_model", "layers"),
            ("model", "language_model", "layers"),
            ("model", "layers"),
            ("language_model", "model", "layers"),
        ]
        layers = None
        for path in candidates:
            obj = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                # sanity: first element has self_attn
                try:
                    if hasattr(obj[0], "self_attn"):
                        layers = obj
                        print(f"  AttentionBlocker: found layers at model.{'.'.join(path)} "
                              f"(n={len(obj)})")
                        break
                except (IndexError, TypeError):
                    continue
        if layers is None:
            raise RuntimeError("could not locate decoder layers on this model")
        self.layers = layers
        self.num_layers = len(self.layers)

    def install(self):
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = attn.register_forward_pre_hook(self._make_hook(i), with_kwargs=True)
            self._hooks.append(h)

    def uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def set_block(self, mask4d: torch.Tensor | None, active_layers: set[int] | None):
        self._mask = mask4d
        self._active_layers = active_layers

    def clear(self):
        self.set_block(None, None)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            if self._mask is None:
                return
            if self._active_layers is not None and layer_idx not in self._active_layers:
                return
            am = kwargs.get("attention_mask", None)
            # handle both prefill (q_len = S) and decode (q_len = 1) shapes
            if am is None:
                kwargs["attention_mask"] = self._mask
                return
            # am: (B, 1, q_len, kv_len). Slice/expand self._mask to match.
            q_len = am.shape[-2]
            kv_len = am.shape[-1]
            m = self._mask
            # m is (1, 1, S, S), built for the prefill length S. At decode time q_len = 1 and
            # kv_len = S + t: slice the last query rows and zero-pad the extra key columns.
            if q_len == m.shape[-2] and kv_len == m.shape[-1]:
                kwargs["attention_mask"] = am + m
            else:
                # build decode-time slice
                m_q = m[..., -q_len:, :]                    # last q_len rows (current queries)
                if kv_len > m.shape[-1]:
                    # newly generated tokens are never blocked
                    extra = kv_len - m.shape[-1]
                    pad = torch.zeros(m_q.shape[0], m_q.shape[1], m_q.shape[2], extra,
                                       device=m_q.device, dtype=m_q.dtype)
                    m_q = torch.cat([m_q, pad], dim=-1)
                elif kv_len < m.shape[-1]:
                    m_q = m_q[..., :kv_len]
                kwargs["attention_mask"] = am + m_q
        return hook


# ── token span helpers ─────────────────────────────────────────────────────
def char_spans_to_token_idx(offset_mapping, char_spans):
    """offset_mapping = list of (char_s, char_e) for each answer/prefix token.
    char_spans = list of (cs, ce).
    Returns set of token indices that overlap any char span.
    """
    out = set()
    for cs, ce in char_spans:
        for i, (s, e) in enumerate(offset_mapping):
            if e <= cs or s >= ce:
                continue
            out.add(i)
    return out


def find_token_at_char(offset_mapping, char_pos: int) -> int:
    """First token index whose span starts at or after ``char_pos``."""
    for i, (s, e) in enumerate(offset_mapping):
        if s >= char_pos:
            return i
    return len(offset_mapping)


# ── build additive mask ────────────────────────────────────────────────────
def build_block_mask(seq_len: int, blocked_keys: set[int], q_start: int,
                     device, dtype) -> torch.Tensor:
    """Return (1, 1, seq_len, seq_len) additive mask: -inf at (q,k) for q >= q_start, k in blocked_keys.
    Otherwise 0.
    """
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    if not blocked_keys or q_start >= seq_len:
        return mask
    blocked = torch.tensor(sorted(blocked_keys), device=device, dtype=torch.long)
    NEG_INF = torch.finfo(dtype).min
    mask[0, 0, q_start:, blocked] = NEG_INF
    return mask


# ── prompt prefix ──────────────────────────────────────────────────────────
def build_prompt_head(processor, enable_thinking: bool = False) -> str:
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": BG_QUESTION},
        ],
    }]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return processor.apply_chat_template(msgs, **kwargs)


# ── offset_mapping helper (fallback for tokenizers without native support) ─
def _encode_with_offsets(tokenizer, text: str):
    """Return (input_ids: list[int], offset_mapping: list[(int,int)]).
    Tries native return_offsets_mapping first; falls back to cumulative-decode
    char alignment for tokenizers that ignore the flag (e.g. some custom
    sentencepiece tokenizers in trust_remote_code models).
    """
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        if "offset_mapping" in enc and enc["offset_mapping"] is not None:
            ids = list(enc["input_ids"])
            offs = [tuple(x) for x in enc["offset_mapping"]]
            if len(offs) == len(ids):
                return ids, offs
    except Exception:
        pass

    # fallback: encode then cumulative-decode to recover char spans.
    enc = tokenizer(text, add_special_tokens=False)
    ids = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = []
    cursor = 0
    prev_decoded = ""
    for i in range(len(ids)):
        cur_decoded = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
        piece = cur_decoded[len(prev_decoded):] if cur_decoded.startswith(prev_decoded) else cur_decoded[len(prev_decoded):]
        prev_decoded = cur_decoded
        if not piece:
            offsets.append((cursor, cursor))
            continue
        idx = text.find(piece, cursor)
        if idx < 0:
            stripped = piece.lstrip()
            if stripped and stripped != piece:
                idx = text.find(stripped, cursor)
                if idx >= 0:
                    offsets.append((idx, idx + len(stripped)))
                    cursor = idx + len(stripped)
                    continue
            # last resort: zero-width span at cursor.
            offsets.append((cursor, cursor))
        else:
            offsets.append((idx, idx + len(piece)))
            cursor = idx + len(piece)
    return ids, offsets


# ── per-image: build inputs + run all conditions ───────────────────────────
def run_image(row: dict, processor, model, tokenizer, blocker: AttentionBlocker,
              prompt_head: str, args) -> list[dict]:
    image_path = row["image"]
    try:
        image = Image.open(image_path).convert("RGB")
        if args.image_resize > 0:
            image = image.resize((args.image_resize, args.image_resize))
    except Exception as e:
        return [{"image": image_path, "error": f"image_open_failed: {e}"}]

    pre_answer = row["pre_answer"]
    full_text = prompt_head + pre_answer

    # Same preprocessing as inference/run_perception.py: resize to 512x512, then
    # process_vision_info with the same question text.
    if process_vision_info is not None:
        vision_msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": BG_QUESTION},
        ]}]
        proc_images, _ = process_vision_info(vision_msgs)
    else:
        proc_images = [image]

    inputs = processor(
        text=[full_text], images=proc_images, padding=True, return_tensors="pt"
    ).to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len > args.max_model_len:
        return [{"image": image_path, "error": f"seq_len_{seq_len}_too_long"}]

    # locate the pre_answer tokens inside the full sequence (they are the trailing tokens)
    pa_ids, pa_offset = _encode_with_offsets(tokenizer, pre_answer)
    n_pa_tokens = len(pa_ids)

    full_ids = inputs["input_ids"][0].tolist()
    pa_start_in_full = seq_len - n_pa_tokens
    # verify the alignment; allow a small shift at the boundary (best effort)
    if pa_start_in_full < 0 or full_ids[pa_start_in_full:] != pa_ids:
        # shift search
        match = False
        for shift in range(-5, 6):
            cand = seq_len - n_pa_tokens + shift
            if cand < 0 or cand + n_pa_tokens > seq_len:
                continue
            if full_ids[cand:cand + n_pa_tokens] == pa_ids:
                pa_start_in_full = cand
                match = True
                break
        if not match:
            return [{"image": image_path, "error": "pa_token_align_failed"}]

    # phrase token positions
    dist_char_spans = [tuple(s) for s in row["dist_spans"]]
    tgt_char_spans = [tuple(s) for s in row["tgt_spans"]]
    dist_tok_in_pa = char_spans_to_token_idx(pa_offset, dist_char_spans)
    tgt_tok_in_pa = char_spans_to_token_idx(pa_offset, tgt_char_spans)
    dist_tok_full = {pa_start_in_full + i for i in dist_tok_in_pa}
    tgt_tok_full = {pa_start_in_full + i for i in tgt_tok_in_pa}

    # qscope 'A': queries from the last '<answer>' sub-token (seq_len-1) onward, plus decode steps
    q_start_A = seq_len - 1

    # num_hidden_layers lives in a different config attribute per model
    L = None
    cfg = model.config
    for path in (("text_config", "num_hidden_layers"),
                 ("num_hidden_layers",),
                 ("llm_config", "num_hidden_layers")):
        obj = cfg
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, int) and obj > 0:
            L = obj
            break
    if L is None:
        # fall back to the layer count discovered by the blocker
        L = blocker.num_layers
    mid_lo = max(0, int(L * 0.5))
    mid_hi = min(L, int(L * 0.8))
    layers_all = None  # None = all
    layers_mid = set(range(mid_lo, mid_hi))

    out_rows = []

    # generation helper
    def greedy_decode(stop_tok_ids):
        """run prefix forward + greedy decode up to args.max_new_tokens or hitting stop."""
        with torch.no_grad():
            out = model(**inputs, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_tok = int(out.logits[0, -1, :].argmax().item())
        generated = [next_tok]
        for step in range(1, args.max_new_tokens):
            if next_tok in stop_tok_ids:
                break
            with torch.no_grad():
                step_inputs = {
                    "input_ids": torch.tensor([[next_tok]], device=model.device),
                    "past_key_values": past,
                    "use_cache": True,
                    "return_dict": True,
                }
                step_out = model(**step_inputs)
                past = step_out.past_key_values
                next_tok = int(step_out.logits[0, -1, :].argmax().item())
            generated.append(next_tok)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text, generated

    answer_close_ids = tokenizer.encode("</answer>", add_special_tokens=False)
    stop_set = set(answer_close_ids)

    # arm → set of blocked key positions
    arm_to_keys = {
        "block_distractor": dist_tok_full,
        "block_target":     tgt_tok_full,
    }

    conditions = [("baseline", None, None)]
    for arm in ("block_distractor", "block_target"):
        for layer_name in ("all", "mid_late"):
            conditions.append((arm, layer_name, "A"))

    for cond in conditions:
        arm, layer_name, qscope = cond
        if arm == "baseline":
            blocker.clear()
            cond_key = "baseline"
            keys_size = 0
        else:
            blocked = arm_to_keys[arm]
            keys_size = len(blocked)
            mask4d = build_block_mask(
                seq_len, blocked, q_start_A, device=model.device, dtype=model.dtype
            )
            active = layers_all if layer_name == "all" else layers_mid
            blocker.set_block(mask4d, active)
            cond_key = f"{arm}__{layer_name}__qscope_{qscope}"

        try:
            cont_text, gen_ids = greedy_decode(stop_set)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            out_rows.append({
                "image": image_path, "label": row["label"], "n": int(row["n"]),
                "condition_key": cond_key, "arm": arm,
                "layer": layer_name, "qscope": qscope,
                "continuation": "", "error": f"OOM: {e}",
            })
            continue
        finally:
            blocker.clear()

        out_rows.append({
            "image": image_path, "label": row["label"], "n": int(row["n"]),
            "condition_key": cond_key,
            "arm": arm, "layer": layer_name, "qscope": qscope,
            "n_dist_tokens": len(dist_tok_full),
            "n_tgt_tokens": len(tgt_tok_full),
            "n_keys_blocked": keys_size,
            "q_start_A": int(q_start_A),
            "seq_len": int(seq_len),
            "continuation": cont_text,
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="out/inputs_qwen3-thinking.parquet")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--class-group", required=True,
                    help="comma-separated target classes, e.g. 'dog,bird,vehicle'")
    ap.add_argument("--filter-n", type=int, default=4)
    ap.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=8,
                    help="force-answer setting: only the class name and </answer> are generated")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--image-resize", type=int, default=512,
                    help="Resize image to NxN at load time. 512 matches inference/run_perception.py; "
                         "use 0 to disable resizing.")
    args = ap.parse_args()

    classes = [c.strip().lower() for c in args.class_group.split(",") if c.strip()]
    if not classes:
        raise SystemExit("[error] --class-group is empty")

    df = pd.read_parquet(args.inputs)
    df = df[df["n"] == args.filter_n]
    df = df[df["label"].str.lower().isin(classes)].reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"▶ {len(df)} images for classes={classes}")

    # skip existing
    done_keys: set[tuple[str, str]] = set()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["image"], r["condition_key"]))
                except Exception:
                    continue
        print(f"  skip-existing: {len(done_keys)} (image, condition) records done")

    print(f"▶ loading processor / model ({args.model_path}) eager attn...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    load_kwargs = dict(
        dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Load any of the four models: AutoModelForImageTextToText, then AutoModelForCausalLM
    # (Intern-S1), then AutoModel.
    model = None
    last_err = None
    loader_candidates = []
    try:
        from transformers import AutoModelForImageTextToText
        loader_candidates.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
    except Exception:
        pass
    try:
        from transformers import AutoModelForCausalLM
        loader_candidates.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    except Exception:
        pass
    try:
        from transformers import AutoModel
        loader_candidates.append(("AutoModel", AutoModel))
    except Exception:
        pass

    for name, cls in loader_candidates:
        try:
            print(f"  trying loader: {name}")
            model = cls.from_pretrained(args.model_path, **load_kwargs)
            print(f"  ✓ loaded via {name}")
            break
        except Exception as e:
            print(f"  × {name} failed: {type(e).__name__}: {str(e)[:200]}")
            last_err = e
            model = None
    if model is None:
        raise SystemExit(f"[error] could not load model {args.model_path}: {last_err}")
    model.eval()

    # num_layers: try the candidate config paths
    num_layers = None
    for path in (("text_config", "num_hidden_layers"),
                 ("num_hidden_layers",),
                 ("llm_config", "num_hidden_layers"),
                 ("text_config", "hidden_size_per_layer_input_dim"),  # safety probe; ignored on KeyError
                 ):
        obj = model.config
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, int) and obj > 0:
            num_layers = obj
            print(f"  num_layers={num_layers} (config.{'.'.join(path)})")
            break
    if num_layers is None:
        print("  ⚠ could not auto-detect num_layers from config; relying on blocker probe.")
    print(f"  ✅ model loaded.")

    blocker = AttentionBlocker(model)
    blocker.install()
    # enable_thinking=True for Intern-S1 only (matches inference/models.py)
    enable_thinking = "intern" in args.model_path.lower()
    prompt_head = build_prompt_head(processor, enable_thinking=enable_thinking)
    print(f"  enable_thinking={enable_thinking}  process_vision_info={'on' if process_vision_info else 'off'}")

    mode_open = "a" if (args.skip_existing and done_keys) else "w"
    fout = open(out_path, mode_open, encoding="utf-8")
    n_done = 0
    n_records = 0
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="block-force"):
            recs = run_image(row.to_dict(), processor, model, tokenizer, blocker,
                             prompt_head, args)
            for r in recs:
                key = (r.get("image"), r.get("condition_key", ""))
                if key in done_keys:
                    continue
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_records += 1
            fout.flush()
            n_done += 1
    finally:
        fout.close()
        blocker.uninstall()
    print(f"[done] images={n_done}  records={n_records}  → {out_path}")


if __name__ == "__main__":
    main()
