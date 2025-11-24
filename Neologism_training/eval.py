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

from qwen_diff_train_forward import qwen_edit_forward

# Import shared utilities from train.py
# (train.py must guard training with if __name__ == "__main__": )
from train import pil_from_bytes, parse_img_field  # noqa: F401


# ---------------------------
# Defaults (match train.py)
# ---------------------------
MODEL_PATH_DEFAULT   = "Qwen/Qwen-Image-Edit"
DATASET_PATH_DEFAULT = "/content/sample_data/filtered_dataset.csv"
SAVE_DIR_DEFAULT     = "./Neologism_training/outputs_neologism_and"

HEIGHT_DEFAULT = 512
WIDTH_DEFAULT  = 512

NUM_STEPS_EVAL_DEFAULT = 15
CFG_EVAL_DEFAULT       = 4.0

DTYPE_DEFAULT = torch.bfloat16  # A100-friendly


# ---------------------------
# Pipeline loader (frozen)
# ---------------------------
def load_frozen_pipeline(model_path: str, dtype: torch.dtype, device: str):
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
    )

    # Offload weights to CPU between calls
    if hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    else:
        # fallback: move only transformer to GPU later
        pass

    # Memory helpers (still useful for runtime)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    try:
        if hasattr(pipe.transformer, "set_attn_processor"):
            pipe.transformer.set_attn_processor("sdpa")
    except Exception:
        pass

    try:
        if hasattr(pipe.transformer, "enable_xformers_memory_efficient_attention"):
            pipe.transformer.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    for module_name in ["text_encoder", "transformer", "vae", "image_encoder"]:
        if hasattr(pipe, module_name):
            for p in getattr(pipe, module_name).parameters():
                p.requires_grad_(False)

    pipe.to(device)  # <-- optional; with offload it won't pin everything anyway
    return pipe


# ---------------------------
# Load neologism ckpt
# ---------------------------
def load_neologism_ckpt(ckpt_path: str, device: str):
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

            # (A) BASELINE
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

            # (B) NEOLOGISM
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

            # Save images + prompt
            base_path = os.path.join(out_dir, f"img_{idx:05d}_base.png")
            neo_path  = os.path.join(out_dir, f"img_{idx:05d}_neo.png")
            cmp_path  = os.path.join(out_dir, f"img_{idx:05d}_cmp.png")
            txt_path  = os.path.join(out_dir, f"img_{idx:05d}.txt")

            img_base.save(base_path)
            img_neo.save(neo_path)

            # side-by-side
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
    print("dtype:", dtype)

    # Load pipeline
    pipeline = load_frozen_pipeline(args.model, dtype, device)

    # Load neologism
    and_neologism_emb, target_token_ids = load_neologism_ckpt(args.neologism_ckpt, device)

    # Load dataset
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
