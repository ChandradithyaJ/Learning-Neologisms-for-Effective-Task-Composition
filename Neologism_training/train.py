import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import io
import gc
import ast
from typing import List
from dotenv import load_dotenv

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImageEditPipeline

from qwen_diff_train_forward import qwen_edit_forward  # your custom forward

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


# ===========================
# 0) Config
# ===========================
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")

MODEL_PATH = "ovedrive/qwen-image-edit-4bit"

# "Neologism" will be tied to the existing word "and"
TARGET_WORDS = [" and"]

LR = 3e-4          # instead of 1e-2
EPOCHS = 1                   # number of passes over df
STEPS_PER_IMAGE = 2          # gradient steps per row

# *** SMALLER SETTINGS FOR TRAINING ***
HEIGHT = 256                 # was 512 — smaller to avoid OOM
WIDTH = 256
NUM_STEPS_TRAIN = 4          # was 8 — fewer steps to save VRAM
NUM_STEPS_EVAL  = 8          # can keep eval a bit higher; no grads
GUIDANCE = 1.0               # true_cfg_scale; 1.0 => no CFG in training
MAX_TRAIN_IMAGES = 2         # cap for debug; set None for full df

# Use reject image as negative example?
USE_REJECT = True # SHOULD BE TRUE FOR TRUE NEOLOGISMS 
REJECT_LAMBDA = 0.05 # weight for pushing away from reject_img

SAVE_DIR = "./Neologism_training/outputs_neologism_and"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ===========================
# 1) Load pipeline (frozen)
# ===========================
pipeline = QwenImageEditPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,   # good for Quadro RTX 6000
)
pipeline.to(device)

# Light memory helpers
if hasattr(pipeline, "enable_attention_slicing"):
    pipeline.enable_attention_slicing("max")
if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
    pipeline.vae.enable_slicing()
if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
    pipeline.vae.enable_tiling()
if hasattr(pipeline, "enable_gradient_checkpointing"):
    pipeline.enable_gradient_checkpointing()

tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
vae = pipeline.vae

# Freeze everything except our neologism embedding
if hasattr(pipeline, "text_encoder"):
    for p in pipeline.text_encoder.parameters():
        p.requires_grad_(False)

if hasattr(pipeline, "transformer"):
    for p in pipeline.transformer.parameters():
        p.requires_grad_(False)

if hasattr(pipeline, "vae"):
    for p in pipeline.vae.parameters():
        p.requires_grad_(False)

if hasattr(pipeline, "image_encoder"):
    for p in pipeline.image_encoder.parameters():
        p.requires_grad_(False)

# ===========================
# 2) Build neologism embedding for "and"
# ===========================
emb_layer = text_encoder.get_input_embeddings()
D = emb_layer.weight.shape[-1]
emb_dtype = emb_layer.weight.dtype
print(f"Text embedding dim: {D}, dtype: {emb_dtype}")

# Get token ids for "and"/"And" (without adding new tokens)
target_token_ids: List[int] = []
with torch.no_grad():
    for w in TARGET_WORDS:
        out = tokenizer(w, add_special_tokens=False)
        ids = out["input_ids"]
        target_token_ids.extend(ids)

target_token_ids = sorted(set(target_token_ids))
print(f"Target token IDs for {TARGET_WORDS}: {target_token_ids}")

# Initialize neologism embedding from the average of these base IDs
with torch.no_grad():
    base_vecs = emb_layer.weight[target_token_ids].to(device=device, dtype=emb_dtype)
    init_vec = base_vecs.mean(dim=0)
    print(f"Initialized 'and' neologism embedding from {len(target_token_ids)} base subword vectors.")

# This is the ONLY learnable parameter: the neologism embedding vector
and_neologism_emb = nn.Parameter(init_vec.clone().detach())
and_neologism_emb.requires_grad_(True)

opt = torch.optim.AdamW([and_neologism_emb], lr=LR)

# ===========================
# 3) Utilities for data & tensors
# ===========================
def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def parse_img_field(value):
    """
    CSV stores things like "{'bytes': b'...'}" as strings.
    This turns them back into dicts with a 'bytes' field.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unexpected type for image field: {type(value)}")

def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Encode a PIL image to the same tensor space the pipeline uses when
    output_type='pt'. Shape: (1, C, H, W).
    """
    img = img.resize((WIDTH, HEIGHT), Image.BICUBIC)

    pixel = pipeline.image_processor.preprocess(img)
    if not isinstance(pixel, torch.Tensor):
        pixel = torch.tensor(pixel)

    # Use same dtype as the main transformer / UNet
    pipe_dtype = next(pipeline.transformer.parameters()).dtype
    pixel = pixel.to(device=device, dtype=pipe_dtype)
    return pixel


def image_to_latents(img: Image.Image) -> torch.Tensor:
    """
    Encode a PIL image to VAE latents compatible with QwenImageEditPipeline.
    Returns latents of shape (1, C, H', W').
    """
    # Resize to match your diffusion resolution
    img = img.resize((WIDTH, HEIGHT), Image.BICUBIC)

    # Use the pipeline's image_processor
    pixel = pipeline.image_processor.preprocess(img)
    if not isinstance(pixel, torch.Tensor):
        pixel = torch.tensor(pixel)

    vae_dtype = next(vae.parameters()).dtype
    pixel = pixel.to(device=device, dtype=vae_dtype)

    # QwenImage VAE expects a frame dimension: (B, C, 1, H, W)
    if pixel.ndim == 4:
        pixel = pixel.unsqueeze(2)

    with torch.no_grad():
        enc = vae.encode(pixel)
        if hasattr(enc, "latent_dist"):
            latents = enc.latent_dist.sample()
        else:
            latents = enc[0] if isinstance(enc, (tuple, list)) else enc

        # Remove dummy frame dimension if present
        if latents.ndim == 5 and latents.shape[2] == 1:
            latents = latents.squeeze(2)

        # Standard SD-style scaling
        latents = latents * 0.18215

    return latents


# ===========================
# 4) Training loop
# ===========================
# ===========================
# 4) Training loop (LATENT LOSS)
# ===========================
def train_on_df(df: pd.DataFrame, epochs: int = EPOCHS, steps_per_image: int = STEPS_PER_IMAGE):
    """
    Train the 'and' neologism embedding on the given dataframe, using *latent loss*.

    Loss ≈ MSE(z_gen, z_target) - λ * MSE(z_gen, z_reject)
    where z_* are VAE latents.
    """
    n_rows = len(df)

    # One-time tokenizer sanity check
    first_prompt = str(df.iloc[0]["instruction"])
    enc = tokenizer(first_prompt, add_special_tokens=True)
    ids = enc["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    print("=== TOKENIZER SANITY CHECK ===")
    print("prompt:", first_prompt)
    print("ids:", ids)
    print("tokens:", toks)
    print("TARGET_WORDS:", TARGET_WORDS)
    print("target_token_ids:", target_token_ids)
    print("================================")

    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")

        for idx, row in df.iterrows():
            try:
                # 1) Decode images + prompt
                src_info = parse_img_field(row["source_img"])
                tgt_info = parse_img_field(row["target_img"])

                src_img = pil_from_bytes(src_info["bytes"])   # stays as PIL for pipeline
                tgt_img = pil_from_bytes(tgt_info["bytes"])
                prompt = str(row["instruction"])

                print(f"Prompt is: {prompt}")

                # 2) Target latents (no grad)
                with torch.no_grad():
                    tgt_lat = image_to_latents(tgt_img).detach()   # (1, C, H', W')

                # 3) Optional reject latents (no grad)
                reject_lat = None
                if USE_REJECT and "reject_img" in df.columns and not pd.isna(row["reject_img"]):
                    rej_info = parse_img_field(row["reject_img"])
                    rej_img = pil_from_bytes(rej_info["bytes"])
                    with torch.no_grad():
                        reject_lat = image_to_latents(rej_img).detach()

                for step in range(steps_per_image):
                    opt.zero_grad(set_to_none=True)

                    # --- Forward with gradients through 'and' embedding ---
                    torch.cuda.empty_cache()

                    with autocast(dtype=torch.float16):
                        out = qwen_edit_forward(
                            pipeline,
                            image=src_img,
                            prompt=prompt,
                            negative_prompt=None,
                            num_inference_steps=NUM_STEPS_TRAIN,   # small, e.g. 1–4
                            true_cfg_scale=GUIDANCE,               # 1.0 => no CFG
                            height=HEIGHT,
                            width=WIDTH,
                            output_type="latent",
                            return_dict=True,
                            and_neologism_emb=and_neologism_emb,
                            target_token_ids=target_token_ids,
                        )
                        gen_lat = out.images  # (1, C, H', W')
                        del out
                        torch.cuda.empty_cache()

                        # Check for NaNs / inf in generated latents
                        if not torch.isfinite(gen_lat).all():
                            print(f"[WARN] Row {idx}, step {step}: non-finite gen_lat; skipping update.")
                            continue

                        # Match dtype & crop
                        gen_lat = gen_lat.to(dtype=tgt_lat.dtype)

                        min_c = min(gen_lat.shape[1], tgt_lat.shape[1])
                        min_h = min(gen_lat.shape[-2], tgt_lat.shape[-2])
                        min_w = min(gen_lat.shape[-1], tgt_lat.shape[-1])

                        gl = gen_lat[:, :min_c, :min_h, :min_w]
                        tl = tgt_lat[:, :min_c, :min_h, :min_w]

                        # Sanity on targets
                        if not torch.isfinite(tl).all():
                            print(f"[WARN] Row {idx}, step {step}: non-finite tgt_lat; skipping.")
                            continue

                        loss_pos = F.mse_loss(gl, tl)
                        del tl

                        # Optional reject latent loss
                        loss_neg = None
                        if USE_REJECT and (reject_lat is not None):
                            rl = reject_lat[:, :min_c, :min_h, :min_w]
                            if torch.isfinite(rl).all():
                                loss_neg = F.mse_loss(gl, rl)
                                loss = loss_pos - REJECT_LAMBDA * loss_neg
                            else:
                                print(f"[WARN] Row {idx}, step {step}: non-finite reject_lat; ignoring reject term.")
                                loss = loss_pos
                            del rl
                        else:
                            loss = loss_pos
                        torch.cuda.empty_cache()

                        # Loss must be finite
                        if not torch.isfinite(loss):
                            print(f"[WARN] Row {idx}, step {step}: non-finite loss; skipping.")
                            continue

                    scaler.scale(loss).backward()
                    # loss.backward()

                    del loss
                    torch.cuda.empty_cache()

                    # Guard gradient on the neologism embedding
                    if and_neologism_emb.grad is None or not torch.isfinite(and_neologism_emb.grad).all():
                        print(f"[WARN] Row {idx}, step {step}: non-finite grad; zeroing & skipping step.")
                        if and_neologism_emb.grad is not None:
                            and_neologism_emb.grad.zero_()
                        continue

                    # Gradient clipping for stability
                    # torch.nn.utils.clip_grad_norm_([and_neologism_emb], max_norm=1.0)

                    # opt.step()

                    # Unscale before gradient clipping
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([and_neologism_emb], max_norm=1.0)
                    
                    # Scaled optimizer step
                    scaler.step(opt)
                    scaler.update()

                    torch.cuda.empty_cache()

                # 5) Logging
                if (idx + 1) % 5 == 0:
                    msg = f"[{idx + 1}/{n_rows}] loss_pos={loss_pos.item():.4f}"
                    if USE_REJECT and (reject_lat is not None) and (loss_neg is not None):
                        msg += f" loss_neg={loss_neg.item():.4f}"
                    msg += f" ||and_emb||={and_neologism_emb.norm().item():.3f}"
                    print(msg)

                # 6) Optional preview
                if (idx + 1) % 50 == 0:
                    with torch.no_grad():
                        out_preview = pipeline(
                            image=src_img,
                            prompt=prompt,
                            num_inference_steps=NUM_STEPS_EVAL,
                            true_cfg_scale=4.0,   # CFG only at eval
                            height=HEIGHT,
                            width=WIDTH,
                            output_type="pil",
                            return_dict=True,
                        )
                        img_preview = out_preview.images[0]
                        img_preview.save(os.path.join(SAVE_DIR, f"train_preview_{idx:05d}.png"))

                # 7) Cleanup
                del tgt_lat, gen_lat
                if reject_lat is not None:
                    del reject_lat
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            except Exception as e:
                print(f"Row {idx} failed during training: {e}")



# ===========================
# 5) Evaluation pass (save outputs)
# ===========================
@torch.no_grad()
def evaluate_and_save(df: pd.DataFrame, subdir: str = "eval"):
    out_dir = os.path.join(SAVE_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in df.iterrows():
        try:
            src_info = parse_img_field(row["source_img"])
            src_img = pil_from_bytes(src_info["bytes"])
            prompt = str(row["instruction"])

            out = pipeline(
                image=src_img,
                prompt=prompt,
                num_inference_steps=NUM_STEPS_EVAL,
                true_cfg_scale=4.0,  # CFG ok here; no grads
                height=HEIGHT,
                width=WIDTH,
                output_type="pil",
                return_dict=True,
            )
            img = out.images[0]
            img.save(os.path.join(out_dir, f"img_{idx:05d}.png"))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Row {idx} failed during eval: {e}")

# ===========================
# 6) Main
# ===========================
if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    required_cols = {"source_img", "target_img", "reject_img", "instruction"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    print(f"Loaded dataset with shape {df.shape}")

    if MAX_TRAIN_IMAGES is not None:
        df = df.head(MAX_TRAIN_IMAGES)
        print(f"Using first {len(df)} rows for training")
    print("Initial neologism norm:", and_neologism_emb.norm().item())

    print("Starting 'and' neologism training...")
    train_on_df(df, epochs=EPOCHS, steps_per_image=STEPS_PER_IMAGE)
    print("Final neologism norm:", and_neologism_emb.norm().item())


    # Optionally bake the learned embedding back into the embedding table
    with torch.no_grad():
        base_ids = list(target_token_ids)
        base_vecs = emb_layer.weight[base_ids]
        print("Original average 'and' embedding norm:",
              base_vecs.mean(dim=0).norm().item())
        # Replace all those rows with the learned vector
        new_vec = and_neologism_emb.to(emb_layer.weight.dtype).to(emb_layer.weight.device)
        for tid in base_ids:
            emb_layer.weight[tid] = new_vec

    # Save the learned neologism embedding
    projector_path = os.path.join(SAVE_DIR, "and_neologism_embedding.pt")
    torch.save(
        {
            "target_words": TARGET_WORDS,
            "token_ids": target_token_ids,
            "embedding": and_neologism_emb.detach().cpu(),
        },
        projector_path,
    )
    print("Saved and_neologism_embedding.pt to", projector_path)

    print("Evaluating with trained 'and' embedding...")
    evaluate_and_save(df, subdir="eval")

    print("Done. Outputs in:", SAVE_DIR)
