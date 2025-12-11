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
import torch.nn.functional as F  # <-- NEW: for cosine similarity
from PIL import Image
from diffusers import QwenImageEditPipeline

from qwen_diff_train_forward import qwen_edit_forward  # your custom forward



MODEL_PATH_DEFAULT   = "Qwen/Qwen-Image-Edit"
DATASET_PATH_DEFAULT = "/content/sample_data/filtered_dataset.csv"
SAVE_DIR_DEFAULT     = "./Neologism_training/outputs_neologism_and"

HEIGHT_DEFAULT = 512
WIDTH_DEFAULT  = 512

NUM_STEPS_EVAL_DEFAULT = 20
CFG_EVAL_DEFAULT       = 4.0

DTYPE_DEFAULT = torch.bfloat16  # A100-friendly


def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def parse_img_field(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unexpected type for image field: {type(value)}")



def load_neologism_ckpt(ckpt_path: str, device: str):
    """
    Load the trained neologism embedding and associated metadata.
    Returns:
        emb_param       : nn.Parameter [hidden_dim], on device (no grad)
        token_ids       : List[int]
        target_words    : Optional[List[str]]
        epoch_idx       : Optional[int]
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    token_ids: List[int] = ckpt.get("token_ids", [])
    target_words = ckpt.get("target_words", None)
    epoch_idx = ckpt.get("epoch", None)
    emb = ckpt["embedding"]  

    
    emb_param = nn.Parameter(emb.float().to(device))
    emb_param.requires_grad_(False)

    print(f"[NEOLOGISM] Loaded from: {ckpt_path}")
    if target_words is not None:
        print(f"[NEOLOGISM] target_words: {target_words}")
    print(f"[NEOLOGISM] token_ids: {token_ids}")
    if epoch_idx is not None:
        print(f"[NEOLOGISM] trained_epoch: {epoch_idx}")
    print(f"[NEOLOGISM] emb norm: {emb_param.norm().item():.6f}")

    return emb_param, token_ids, target_words, epoch_idx


@torch.no_grad()
def generate_synonyms_with_lm(
    pipeline,
    word_str: str,
    device: str,
    override_emb: Optional[torch.Tensor] = None,
    token_ids: Optional[List[int]] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Use the underlying Qwen2.5-VL LM to answer:
        "Give me 5 synonyms for the word {word_str}"

    If override_emb and token_ids are provided, we temporarily patch those
    token IDs in the LM's embedding table with override_emb during generation,
    then restore the originals afterward.
    """
    lm = pipeline.text_encoder  # Qwen2_5_VLForConditionalGeneration
    tok = pipeline.tokenizer
    lm.eval()

    emb = lm.get_input_embeddings()
    weight = emb.weight

    backup_vecs = None
    if override_emb is not None and token_ids is not None and len(token_ids) > 0:
        backup_vecs = weight[token_ids].clone()
        neo_vec = override_emb.to(weight.device).to(weight.dtype)
        for tid in token_ids:
            weight[tid] = neo_vec

    prompt = (
        f"Give me 5 synonyms for the word {word_str}.\n"
        "Synonyms:"
    )
    inputs = tok(prompt, return_tensors="pt").to(device)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    if tok.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tok.eos_token_id

    outputs = lm.generate(**inputs, **generate_kwargs)
    text = tok.decode(outputs[0], skip_special_tokens=True)

    if backup_vecs is not None:
        for i, tid in enumerate(token_ids):
            weight[tid] = backup_vecs[i]

    return text



@torch.no_grad()
def analyze_neologism_and_save(
    pipeline,
    and_neologism_emb: torch.Tensor,
    target_token_ids: List[int],
    target_words,
    epoch_idx: Optional[int],
    save_dir: str,
):
    """
    1) Compute cosine similarity + L2 distance between:
         - neologism embedding
         - original 'and' embedding (mean over token_ids)
    2) Ask LM for synonyms:
         - baseline 'and' (no override)
         - neologism version (temporary override)
    3) Write everything to a text file in save_dir.
    """
    lm = pipeline.text_encoder
    tok = pipeline.tokenizer

    # get original embeddings for the token IDs (Qwen base weights)
    emb_layer = lm.get_input_embeddings()
    weight = emb_layer.weight
    orig_vecs = weight[target_token_ids]  
    orig_mean = orig_vecs.mean(dim=0)

    neo_vec = and_neologism_emb.detach().to(orig_mean.device).to(orig_mean.dtype)

    cos_sim = F.cosine_similarity(neo_vec.unsqueeze(0), orig_mean.unsqueeze(0)).item()
    l2_dist = torch.norm(neo_vec - orig_mean).item()

    baseline_surface = "and"
    baseline_syn = generate_synonyms_with_lm(
        pipeline=pipeline,
        word_str=baseline_surface,
        device=next(lm.parameters()).device,
        override_emb=None,
        token_ids=None,
    )


    if target_words is not None and len(target_words) > 0:
        neologism_surface = target_words[0]
    else:
        neologism_surface = " and"

    neo_syn = generate_synonyms_with_lm(
        pipeline=pipeline,
        word_str=neologism_surface,
        device=next(lm.parameters()).device,
        override_emb=and_neologism_emb,
        token_ids=target_token_ids,
    )


    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "neologism_analysis.txt")
    with open(out_path, "w") as f:
        f.write("=== Neologism vs Original 'and' Analysis ===\n\n")
        f.write(f"Checkpoint path: {os.path.abspath(save_dir)}\n")
        if epoch_idx is not None:
            f.write(f"Neologism trained epoch: {epoch_idx}\n")
        f.write(f"Target words: {target_words}\n")
        f.write(f"Target token IDs: {target_token_ids}\n\n")

        f.write("Cosine similarity (neologism vs original 'and' embedding mean): "
                f"{cos_sim:.6f}\n")
        f.write("L2 distance      (neologism vs original 'and' embedding mean): "
                f"{l2_dist:.6f}\n\n")

        f.write("----- Baseline LM (original 'and') -----\n")
        f.write('Prompt: "Give me 5 synonyms for the word and."\n')
        f.write("Response:\n")
        f.write(baseline_syn)
        f.write("\n\n")

        f.write("----- LM with Neologism Embedding -----\n")
        f.write(f'Prompt: "Give me 5 synonyms for the word {neologism_surface}."\n')
        f.write("Response:\n")
        f.write(neo_syn)
        f.write("\n")

    print(f"[ANALYSIS] Saved neologism_analysis.txt to: {out_path}")



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


            base_path = os.path.join(out_dir, f"img_{idx:05d}_base.png")
            neo_path  = os.path.join(out_dir, f"img_{idx:05d}_neo.png")
            cmp_path  = os.path.join(out_dir, f"img_{idx:05d}_cmp.png")
            txt_path  = os.path.join(out_dir, f"img_{idx:05d}.txt")

            img_base.save(base_path)
            img_neo.save(neo_path)

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

    pipeline = QwenImageEditPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    print(pipeline)

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("max")
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()

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


    for module_name in ["text_encoder", "transformer", "vae", "image_encoder"]:
        if hasattr(pipeline, module_name):
            for p in getattr(pipeline, module_name).parameters():
                p.requires_grad_(False)


    and_neologism_emb, target_token_ids, target_words, epoch_idx = load_neologism_ckpt(
        args.neologism_ckpt,
        device,
    )


    analyze_neologism_and_save(
        pipeline=pipeline,
        and_neologism_emb=and_neologism_emb,
        target_token_ids=target_token_ids,
        target_words=target_words,
        epoch_idx=epoch_idx,
        save_dir=args.save_dir,
    )

    #Load data
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
