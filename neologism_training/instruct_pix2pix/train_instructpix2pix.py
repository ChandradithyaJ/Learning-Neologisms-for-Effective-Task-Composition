"""
Train neologism embedding for AND
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
from instruct_pix2pix_diff_forward import generate_with_gradients, decode_latents_to_image, latents_to_tensor
from utils.neologism_utils import ClipModel, pil_from_bytes, parse_img_field
import pandas as pd
import gc
import time

DATASET_PATH = "../../scratch/DL_data/filtered_dataset.csv"
TARGET_WORDS = [" and"]

# Learning / schedule
LR = 1e-4
EPOCHS = 20
STEPS_PER_IMAGE = 1

# Resolution
HEIGHT = 512
WIDTH  = 512

# Denoising
NUM_STEPS_TRAIN = 8   # training steps for instruct_pix2pix
NUM_STEPS_EVAL  = 15  # nicer previews (no grads)
GUIDANCE = 1.0        # no CFG during training (true_cfg_scale)   

MAX_TRAIN_IMAGES = 80

# Reject loss
USE_REJECT = False
REJECT_LAMBDA = 0.00

SAVE_DIR = f"./instruct_pix2pix/results/instruct_pix2pix_outputs_neologism_and_{STEPS_PER_IMAGE}stepsPerImage_{MAX_TRAIN_IMAGES}trainImages"
SAVE_EVERY_N = 1
os.makedirs(SAVE_DIR, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# datatype
DTYPE = torch.float16
print("Using dtype:", DTYPE)

def save_neologism_checkpoint(epoch_idx, target_token_ids, and_neologism_emb):
    """
    Save the current neologism embedding (and metadata).
    epoch_idx is 0-based; we’ll save using 1-based numbering in filename.
    """
    ckpt_path = os.path.join(SAVE_DIR, f"and_neologism_epoch_{epoch_idx+1:03d}.pt")
    torch.save(
        {
            "target_words": TARGET_WORDS,
            "token_ids": target_token_ids,
            "embedding": and_neologism_emb.detach().cpu(),
            "epoch": epoch_idx + 1,
        },
        ckpt_path,
    )
    print(f"[SAVE] Neologism checkpoint saved to {ckpt_path}")

def neologism_distance_stats(and_neologism_emb, orig_and_emb):
    cur = and_neologism_emb.detach()
    orig = orig_and_emb
    l2_dist = torch.norm(cur - orig).item()
    cosine_sim = F.cosine_similarity(cur.unsqueeze(0), orig.unsqueeze(0)).item()
    return l2_dist, cosine_sim

if __name__ == "__main__":
    model_id = "timbrooks/instruct-pix2pix"
    finetuned_model_id = "vinesmsuic/magicbrush-jul7"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        finetuned_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.unet.enable_gradient_checkpointing()
    pipe.vae.enable_gradient_checkpointing()
    if hasattr(pipe.text_encoder, 'gradient_checkpointing_enable'):
        pipe.text_encoder.gradient_checkpointing_enable()

    try:
        from diffusers.utils import is_xformers_available
        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
    except:
        print("xformers not available, using default attention")

    pipe.enable_attention_slicing(1)
    pipe.vae.enable_slicing()

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    # Freeze everything except our neologism embedding
    modules = ["text_encoder", "vae", "unet"]
    for module_name in modules:
        if hasattr(pipe, module_name):
            for p in getattr(pipe, module_name).parameters():
                p.requires_grad_(False)

    # Build neologism embedding for "and"
    emb_layer = text_encoder.get_input_embeddings()
    emb_dtype = emb_layer.weight.dtype
    print(f"Text embedding dim: {emb_layer.weight.shape[-1]}, dtype: {emb_dtype}")

    target_token_ids = []
    with torch.no_grad():
        for w in TARGET_WORDS:
            out = tokenizer(w, add_special_tokens=False)
            target_token_ids.extend(out["input_ids"])
    target_token_ids = sorted(set(target_token_ids))
    print(f"Target token IDs for {TARGET_WORDS}: {target_token_ids}")

    with torch.no_grad():
        base_vecs = emb_layer.weight[target_token_ids].to(device=device, dtype=emb_dtype)
        init_vec = base_vecs.mean(dim=0)
        orig_and_emb = init_vec.float().detach().clone().to(device)
        print(f"Initialized neologism from {len(target_token_ids)} base vectors.")

    and_neologism_emb = nn.Parameter(init_vec.float().clone().detach()).to(device)
    and_neologism_emb.requires_grad_(True)
    optimizer = torch.optim.AdamW([and_neologism_emb], lr=LR)

    # Training Loop (CLIP + reject loss)
    dataset = pd.read_csv(DATASET_PATH)
    n_rows = len(dataset)
    dataset = dataset.head(min(n_rows, MAX_TRAIN_IMAGES))

    # tokenizer sanity check
    first_prompt = str(dataset.iloc[0]["instruction"])
    enc = tokenizer(first_prompt, add_special_tokens=True)
    print("=== TOKENIZER SANITY CHECK ===")
    print("prompt:", first_prompt)
    print("ids:", enc["input_ids"])
    print("tokens:", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
    print("target_token_ids:", target_token_ids)
    print("================================")

    # We'll use CLIP image embeddings and cosine similarity.
    clip = ClipModel(device=device)

    for epoch in range(EPOCHS):
        print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")
        start_time = time.time()
        epoch_loss_pos_sum = 0.0
        epoch_loss_neg_sum = 0.0
        epoch_num_pos = 0
        epoch_num_neg = 0

        for idx, row in dataset.iterrows():
            loss_pos = None
            loss_neg = None

            src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
            tgt_img = pil_from_bytes(parse_img_field(row["target_img"])["bytes"])
            prompt  = str(row["instruction"])

            # Precompute target CLIP embedding (no grad needed)
            tgt_clip = clip.clip_encode_from_pil(tgt_img)

            reject_clip = None
            if USE_REJECT and "reject_img" in dataset.columns and not pd.isna(row["reject_img"]):
                rej_img = pil_from_bytes(parse_img_field(row["reject_img"])["bytes"])
                reject_clip = clip.clip_encode_from_pil(rej_img)

            for step in range(STEPS_PER_IMAGE):
                optimizer.zero_grad(set_to_none=True)

                gen_lat = generate_with_gradients(
                    pipe,
                    src_img,
                    prompt,
                    num_inference_steps=NUM_STEPS_TRAIN, # or 30
                    guidance_scale=7.5,
                    image_guidance_scale=1.5,
                    neologism_emb=and_neologism_emb,
                    target_token_ids=target_token_ids
                )

                gen_lat = latents_to_tensor(pipe, gen_lat)
                torch.cuda.empty_cache()

                if not torch.isfinite(gen_lat).all():
                    print(f"[WARN] Row {idx}, step {step}: non-finite gen_lat; skipping.")
                    continue

                # gen_img = decode_latents_to_image(pipe, gen_lat)

                gen_clip = clip.clip_encode_from_tensor(gen_lat)

                del gen_lat
                torch.cuda.empty_cache()

                # CLIP cosine losses (positive + reject)
                loss_pos = 1.0 - F.cosine_similarity(gen_clip, tgt_clip, dim=-1).mean()

                if USE_REJECT and reject_clip is not None:
                    loss_neg = 1.0 - F.cosine_similarity(gen_clip, reject_clip, dim=-1).mean()
                    loss = loss_pos - REJECT_LAMBDA * loss_neg
                else:
                    loss = loss_pos

                if not torch.isfinite(loss):
                    print(f"[WARN] Row {idx}, step {step}: non-finite loss; skipping.")
                    continue

                del gen_clip
                torch.cuda.empty_cache()

                # LOGGING (per-step accumulation)
                epoch_loss_pos_sum += loss_pos.item()
                epoch_num_pos += 1
                if USE_REJECT and reject_clip is not None and loss_neg is not None:
                    epoch_loss_neg_sum += loss_neg.item()
                    epoch_num_neg += 1

                # Backprop only wrt and_neologism_emb
                loss.backward()
                torch.nn.utils.clip_grad_norm_([and_neologism_emb], 1.0)
                optimizer.step()
                torch.cuda.empty_cache()

            if (idx + 1) % 5 == 0 and loss_pos is not None:
                msg = f"[{idx+1}/{MAX_TRAIN_IMAGES}] loss_pos={loss_pos.item():.4f}"
                if USE_REJECT and loss_neg is not None:
                    msg += f" loss_neg={loss_neg.item():.4f}"
                msg += f" ||and_emb||={and_neologism_emb.norm().item():.8f}"
                print(msg)

                torch.cuda.empty_cache()
                gc.collect()

            # Aggressive cleanup after each step
            del loss, loss_pos
            if USE_REJECT and 'loss_neg' in locals():
                del loss_neg

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Time elapsed for epoch {epoch}: {time.time()-start_time}s")

        # Epoch-wise averages
        if epoch_num_pos > 0:
            avg_pos = epoch_loss_pos_sum / epoch_num_pos
        else:
            avg_pos = float("nan")

        if epoch_num_neg > 0:
            avg_neg = epoch_loss_neg_sum / epoch_num_neg
        else:
            avg_neg = float("nan")

        print(f"[EPOCH {epoch+1}] avg_pos_loss={avg_pos:.6f}, avg_neg_loss={avg_neg:.6f} "
              f"(pos_steps={epoch_num_pos}, neg_steps={epoch_num_neg})")

        if (epoch + 1) % SAVE_EVERY_N == 0:
            save_neologism_checkpoint(epoch, target_token_ids, and_neologism_emb)
            l2_dist, cos_sim = neologism_distance_stats(and_neologism_emb, orig_and_emb)
            print(f"[CHECKPOINT EPOCH {epoch+1}] L2={l2_dist:.6f} | cosine={cos_sim:.6f}")

            with open(os.path.join(SAVE_DIR, "neologism_drift_log.txt"), "a") as f:
                f.write(f"epoch={epoch+1:03d}  L2={l2_dist:.6f}  cosine={cos_sim:.6f}\n")

            with open(os.path.join(SAVE_DIR, "epoch_loss_log.txt"), "a") as f:
                f.write(
                    f"epoch={epoch+1:03d}  avg_pos_loss={avg_pos:.6f}  "
                    f"avg_neg_loss={avg_neg:.6f}  pos_steps={epoch_num_pos}  "
                    f"neg_steps={epoch_num_neg}\n"
                )