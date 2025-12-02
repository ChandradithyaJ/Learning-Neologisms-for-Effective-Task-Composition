# eval.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import gc
import ast
import argparse
from typing import List, Optional

from dotenv import load_dotenv
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from diffusers import QwenImageEditPipeline

from qwen_diff_train_forward import qwen_edit_forward  # your custom forward


# ---------------------------
# Defaults (match train.py where relevant)
# ---------------------------
MODEL_PATH_DEFAULT   = "Qwen/Qwen-Image-Edit"
DATASET_PATH_DEFAULT = "/content/sample_data/filtered_dataset.csv"
SAVE_DIR_DEFAULT     = "./Neologism_training/outputs_neologism_and"

HEIGHT_DEFAULT = 512
WIDTH_DEFAULT  = 512

NUM_STEPS_EVAL_DEFAULT = 10
CFG_EVAL_DEFAULT       = 4.0

DTYPE_DEFAULT = torch.bfloat16  # A100-friendly


# ---------------------------
# Simple shared utilities
# (avoid importing train.py to prevent double-loading the pipeline)
# ---------------------------
def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def parse_img_field(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unexpected type for image field: {type(value)}")


# ---------------------------
# Load neologism ckpt
# ---------------------------
def load_neologism_ckpt(ckpt_path: str, device: str):
    """
    Load the trained neologism embedding and associated metadata.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    token_ids: List[int] = ckpt.get("token_ids", [])
    target_words = ckpt.get("target_words", None)
    emb = ckpt["embedding"]  # Tensor [hidden_dim]

    # put on device in fp32 (matches train)
    emb_param = nn.Parameter(emb.float().to(device))
    emb_param.requires_grad_(False)

    print(f"[NEOLOGISM] Loaded from: {ckpt_path}")
    if target_words is not None:
        print(f"[NEOLOGISM] target_words: {target_words}")
    print(f"[NEOLOGISM] token_ids: {token_ids}")
    print(f"[NEOLOGISM] emb norm: {emb_param.norm().item():.6f}")

    return emb_param, token_ids


# ---------------------------
# Eval + save (baseline vs neo)
# ---------------------------
@torch.no_grad()
def evaluate_and_save(
    df: pd.DataFrame,
    pipeline,
    and_neologism_emb: Optional[torch.Tensor],
    target_token_ids: Optional[List[int]],
    save_dir: str,
    subdir: str = "eval",
    height: int = HEIGHT_DEFAULT,
    width: int = WIDTH_DEFAULT,
    num_steps_eval: int = NUM_STEPS_EVAL_DEFAULT,
    cfg_eval: float = CFG_EVAL_DEFAULT,
):
    out_dir = os.path.join(save_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in df.iterrows():
        try:
            src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
            prompt  = str(row["instruction"])

            # ---------
            # (A) BASELINE: no neologism injection
            # ---------
            out_base = qwen_edit_forward(
                pipeline,
                image=src_img,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=num_steps_eval,
                true_cfg_scale=cfg_eval,
                height=height,
                width=width,
                output_type="pil",
                return_dict=True,
                and_neologism_emb=None,
                target_token_ids=None,
            )
            img_base = out_base.images[0]

            # ---------
            # (B) NEOLOGISM: inject trained embedding
            # ---------
            out_neo = qwen_edit_forward(
                pipeline,
                image=src_img,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=num_steps_eval,
                true_cfg_scale=cfg_eval,
                height=height,
                width=width,
                output_type="pil",
                return_dict=True,
                and_neologism_emb=and_neologism_emb,
                target_token_ids=target_token_ids,
            )
            img_neo = out_neo.images[0]

            # ---------
            # Save images + prompt
            # ---------
            base_path = os.path.join(out_dir, f"img_{idx:05d}_base.png")
            neo_path  = os.path.join(out_dir, f"img_{idx:05d}_neo.png")
            cmp_path  = os.path.join(out_dir, f"img_{idx:05d}_cmp.png")
            txt_path  = os.path.join(out_dir, f"img_{idx:05d}.txt")

            img_base.save(base_path)
            img_neo.save(neo_path)

            # side-by-side comparison
            w, h = img_base.size
            cmp = Image.new("RGB", (w * 2, h))
            cmp.paste(img_base, (0, 0))
            cmp.paste(img_neo,  (w, 0))
            cmp.save(cmp_path)

            with open(txt_path, "w") as f:
                f.write(prompt)

            print(f"[EVAL] saved {idx:05d}: base / neo / cmp")

        except Exception as e:
            print(f"Eval row {idx} failed: {e}")

        # light cleanup; remove if you want max speed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET_PATH_DEFAULT)
    parser.add_argument("--model", type=str, default=MODEL_PATH_DEFAULT)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR_DEFAULT)
    parser.add_argument(
        "--neologism_ckpt",
        type=str,
        default=os.path.join(SAVE_DIR_DEFAULT, "and_neologism_embedding.pt"),
        help="Path to saved neologism checkpoint (.pt).",
    )
    parser.add_argument("--subdir", type=str, default="eval")
    parser.add_argument("--height", type=int, default=HEIGHT_DEFAULT)
    parser.add_argument("--width", type=int, default=WIDTH_DEFAULT)
    parser.add_argument("--num_steps_eval", type=int, default=NUM_STEPS_EVAL_DEFAULT)
    parser.add_argument("--cfg_eval", type=float, default=CFG_EVAL_DEFAULT)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_DEFAULT
    print("device:", device)
    print("Using dtype:", dtype)

    # 1) Load pipeline — like training, but only once
    pipeline = QwenImageEditPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)

    # Light memory helpers (same style as training)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("max")
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()

    # Efficient attention / xFormers
    try:
        if hasattr(pipeline.transformer, "set_attn_processor"):
            pipeline.transformer.set_attn_processor("sdpa")
    except Exception:
        pass

    try:
        if hasattr(pipeline.transformer, "enable_xformers_memory_efficient_attention"):
            pipeline.transformer.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Freeze everything (no grads in eval)
    for module_name in ["text_encoder", "transformer", "vae", "image_encoder"]:
        if hasattr(pipeline, module_name):
            for p in getattr(pipeline, module_name).parameters():
                p.requires_grad_(False)

    # 2) Load neologism emb
    and_neologism_emb, target_token_ids = load_neologism_ckpt(args.neologism_ckpt, device)

    # 3) Load dataset
    df = pd.read_csv(args.dataset)
    required_cols = {"source_img", "instruction"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    if args.max_images is not None:
        df = df.head(args.max_images)
        print(f"Using first {len(df)} rows for eval")

    print(f"Evaluating on dataset with shape {df.shape}...")
    evaluate_and_save(
        df=df,
        pipeline=pipeline,
        and_neologism_emb=and_neologism_emb,
        target_token_ids=target_token_ids,
        save_dir=args.save_dir,
        subdir=args.subdir,
        height=args.height,
        width=args.width,
        num_steps_eval=args.num_steps_eval,
        cfg_eval=args.cfg_eval,
    )

    print("Done. Outputs in:", os.path.join(args.save_dir, args.subdir))


if __name__ == "__main__":
    main()
