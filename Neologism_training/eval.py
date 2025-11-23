#!/usr/bin/env python3
"""
eval.py

Example:
python eval.py \
  --df_path /content/sample_data/filtered_dataset.csv \
  --neo_pt /content/Multitask-Image-Editing-via-Neologisms-and-Textual-Inversion/training-attempt1-results/and_neologism_epoch_010.pt \
  --model_path Qwen/Qwen-Image-Edit \
  --out_dir /content/sample_data/Neo_res \
  --num_steps 15 \
  --true_cfg 4.0 \
  --max_rows 50
"""

import os
import argparse
from typing import Optional, Tuple, Any

import pandas as pd
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# reuse your forward
from qwen_diff_train_forward import qwen_edit_forward


# -------------------------
# Defaults you can edit
# -------------------------
DEFAULT_IMAGE_COL = "image_path"
DEFAULT_PROMPT_COL = "prompt"
DEFAULT_NEG_PROMPT_COL = "negative_prompt"


def load_neologism(pt_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Loads a neologism embedding saved as a .pt.
    Supports:
      - raw tensor
      - dict with common keys
    """
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        emb = obj
    elif isinstance(obj, dict):
        # try a few likely keys
        for k in ["and_neologism_emb", "embedding", "emb", "vec", "tensor"]:
            if k in obj:
                emb = obj[k]
                break
        else:
            raise ValueError(f"Dict in {pt_path} had no recognized embedding key. Keys: {list(obj.keys())}")
    else:
        raise ValueError(f"Unsupported neologism format: {type(obj)}")

    emb = emb.to(device=device, dtype=dtype)
    if emb.ndim != 1:
        emb = emb.view(-1)  # flatten just in case
    return emb


def get_target_token_ids(pipe: QwenImageEditPipeline, target_str: str = " and") -> list:
    """
    Compute token id(s) for the target string once.
    We set add_special_tokens=False to get just the lexical tokens.
    """
    enc = pipe.tokenizer(target_str, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    return ids


def load_image(path_or_obj: Any) -> Image.Image:
    """
    If dataframe stores paths, load them.
    If it stores PIL directly, just return it.
    """
    if isinstance(path_or_obj, Image.Image):
        return path_or_obj
    if not isinstance(path_or_obj, str):
        raise ValueError(f"image entry must be a path or PIL.Image, got {type(path_or_obj)}")
    return Image.open(path_or_obj).convert("RGB")


@torch.inference_mode(False)
def run_one(
    pipe: QwenImageEditPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: Optional[str],
    num_steps: int,
    true_cfg: float,
    guidance_scale: Optional[float],
    output_type: str,
    neo_emb: Optional[torch.Tensor],
    target_token_ids: Optional[list],
    seed: Optional[int],
) -> Image.Image:
    """
    Runs your differentiable forward once.
    We keep it in autocast for speed/memory.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = qwen_edit_forward(
            pipe=pipe,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            true_cfg_scale=true_cfg,
            guidance_scale=guidance_scale,
            output_type=output_type,
            return_dict=True,
            # neologism injection
            and_neologism_emb=neo_emb,
            target_token_ids=target_token_ids,
        )
    # out.images is PIL list or tensor depending on output_type
    imgs = out.images
    if isinstance(imgs, list):
        return imgs[0]
    return imgs  # latent or tensor path if you ever switch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", required=True, help="CSV or Parquet dataframe path")
    parser.add_argument("--neo_pt", required=True, help="Path to neologism .pt embedding")
    parser.add_argument("--model_path", default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--image_col", default=DEFAULT_IMAGE_COL)
    parser.add_argument("--prompt_col", default=DEFAULT_PROMPT_COL)
    parser.add_argument("--neg_prompt_col", default=DEFAULT_NEG_PROMPT_COL)

    parser.add_argument("--num_steps", type=int, default=15)
    parser.add_argument("--true_cfg", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=None)

    parser.add_argument("--output_type", default="pil", choices=["pil", "latent"])
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_dir = os.path.join(args.out_dir, "baseline")
    neo_dir = os.path.join(args.out_dir, "neologism")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(neo_dir, exist_ok=True)

    # ---- load df ----
    if args.df_path.endswith(".parquet"):
        df = pd.read_parquet(args.df_path)
    else:
        df = pd.read_csv(args.df_path)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()

    # ---- load pipeline ----
    pipe = QwenImageEditPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    device = pipe._execution_device
    dtype = torch.bfloat16

    # ---- get neologism + target token ids ----
    neo_emb = load_neologism(args.neo_pt, device=device, dtype=dtype)
    target_token_ids = get_target_token_ids(pipe, target_str=" and")

    # ---- run ----
    results = []
    for idx, row in df.iterrows():
        try:
            image = load_image(row[args.image_col])
            prompt = row[args.prompt_col]

            neg_prompt = None
            if args.neg_prompt_col in row and pd.notna(row[args.neg_prompt_col]):
                neg_prompt = str(row[args.neg_prompt_col])

            # baseline
            base_img = run_one(
                pipe, image, prompt, neg_prompt,
                num_steps=args.num_steps,
                true_cfg=args.true_cfg,
                guidance_scale=args.guidance_scale,
                output_type=args.output_type,
                neo_emb=None,
                target_token_ids=None,
                seed=args.seed,
            )

            # neologism
            neo_img = run_one(
                pipe, image, prompt, neg_prompt,
                num_steps=args.num_steps,
                true_cfg=args.true_cfg,
                guidance_scale=args.guidance_scale,
                output_type=args.output_type,
                neo_emb=neo_emb,
                target_token_ids=target_token_ids,
                seed=args.seed,
            )

            if args.output_type == "pil":
                base_path = os.path.join(base_dir, f"{idx:06d}.png")
                neo_path  = os.path.join(neo_dir,  f"{idx:06d}.png")
                base_img.save(base_path)
                neo_img.save(neo_path)
            else:
                base_path = os.path.join(base_dir, f"{idx:06d}.pt")
                neo_path  = os.path.join(neo_dir,  f"{idx:06d}.pt")
                torch.save(base_img.detach().cpu(), base_path)
                torch.save(neo_img.detach().cpu(), neo_path)

            results.append({
                "row_idx": idx,
                "baseline_out": base_path,
                "neologism_out": neo_path,
                "prompt": prompt,
                "negative_prompt": neg_prompt,
            })

            if (len(results) % args.save_every) == 0:
                print(f"[{len(results)}/{len(df)}] saved {idx}")

        except Exception as e:
            print(f"[ERROR] row {idx} failed: {e}")
            results.append({"row_idx": idx, "error": str(e)})

    # save a manifest for convenience
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(results).to_csv(manifest_path, index=False)
    print("Done. Manifest:", manifest_path)


if __name__ == "__main__":
    main()
