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
from instruct_pix2pix_diff_forward import generate_with_gradients_lang
from utils.neologism_utils import ClipModel
import pandas as pd
import gc
import time

DATASET_PATH = "../../scratch/DL_data/processed_lang_dataset.parquet"
TARGET_WORDS = [" and"]

# Learning / schedule
LR = 1e-4
EPOCHS = 100 

MAX_TRAIN_EXAMPLES = 100

# Reject loss
USE_REJECT = False
REJECT_LAMBDA = 0.00

SAVE_DIR = f"./instruct_pix2pix/results/instruct_pix2pix_outputs_neologism_and_{MAX_TRAIN_EXAMPLES}trainExamples_{EPOCHS}epochs_LANG_temp"
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
    dataset = pd.read_parquet(DATASET_PATH)
    n_rows = len(dataset)
    dataset = dataset.head(min(n_rows, MAX_TRAIN_EXAMPLES))

    # tokenizer sanity check
    first_prompt = str(dataset.iloc[0]["question"])
    enc = tokenizer(first_prompt, add_special_tokens=True)
    print("=== TOKENIZER SANITY CHECK ===")
    print("prompt:", first_prompt)
    print("ids:", enc["input_ids"])
    print("tokens:", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
    print("target_token_ids:", target_token_ids)
    print("================================")

    # We'll use CLIP text embeddings and cosine similarity.
    clip = ClipModel(device=device)

    for epoch in range(EPOCHS):
        print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")
        start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_num = 0

        for idx, row in dataset.iterrows():
            loss_pos = None
            loss_neg = None

            pref_response = str(row['final_answer'])
            unpref_response = str(row['subanswers'].split('|')[0])
            prompt = str(" and ".join(row["subquestions"].split('|')))

            optimizer.zero_grad(set_to_none=True)

            prompt_embeds = generate_with_gradients_lang(
                pipe,
                prompt,
                neologism_emb=and_neologism_emb,
                target_token_ids=target_token_ids
            )
            torch.cuda.empty_cache()

            if not torch.isfinite(prompt_embeds).all():
                print(f"[WARN] Row {idx}, step {step}: non-finite prompt_embeds; skipping.")
                continue

            pref_embeds = clip.clip_encode_text(pref_response)
            unpref_embeds = clip.clip_encode_text(unpref_response)

            # mean pooling
            print(prompt_embeds.shape, pref_embeds.shape, unpref_embeds.shape)
            prompt_embeds = prompt_embeds.mean(dim=1)    # [1, 768]
            pref_embeds = pref_embeds.mean(dim=1)
            unpref_embeds = unpref_embeds.mean(dim=1)

            # Contrastive Preference Loss with a dot product scorer
            s_pos = (prompt_embeds * pref_embeds).sum(dim=-1)
            s_neg = (prompt_embeds * unpref_embeds).sum(dim=-1)
            print(prompt_embeds, s_pos, s_neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()

            print(loss.item())

            if not torch.isfinite(loss):
                print(f"[WARN] Row {idx}, step {step}: non-finite loss; skipping.")
                continue

            del prompt_embeds, pref_embeds, unpref_embeds
            torch.cuda.empty_cache()

            # LOGGING (per-step accumulation)
            epoch_loss_sum += loss.item()

            # Backprop only wrt and_neologism_emb
            loss.backward()
            torch.nn.utils.clip_grad_norm_([and_neologism_emb], 1.0)
            optimizer.step()
            torch.cuda.empty_cache()

        if (idx + 1) % 5 == 0 and loss is not None:
            msg = f"[{idx+1}/{MAX_TRAIN_EXAMPLES}] loss_pos={loss.item():.4f}"
            msg += f" ||and_emb||={and_neologism_emb.norm().item():.8f}"
            print(msg)

            # Aggressive cleanup after each step
            del loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Time elapsed for epoch {epoch}: {time.time()-start_time}s")

        # Epoch-wise averages
        if epoch_num > 0:
            avg_loss = epoch_loss_sum / epoch_num
        else:
            avg_loss = float("nan")

        print(f"[EPOCH {epoch+1}] avg_loss={avg_loss:.6f}"
              f"(pos_steps={epoch_num}")

        if (epoch + 1) % SAVE_EVERY_N == 0:
            save_neologism_checkpoint(epoch, target_token_ids, and_neologism_emb)
            l2_dist, cos_sim = neologism_distance_stats(and_neologism_emb, orig_and_emb)
            print(f"[CHECKPOINT EPOCH {epoch+1}] L2={l2_dist:.6f} | cosine={cos_sim:.6f}")

            with open(os.path.join(SAVE_DIR, "neologism_drift_log.txt"), "a") as f:
                f.write(f"epoch={epoch+1:03d}  L2={l2_dist:.6f}  cosine={cos_sim:.6f}\n")

            with open(os.path.join(SAVE_DIR, "epoch_loss_log.txt"), "a") as f:
                f.write(
                    f"epoch={epoch+1:03d}  avg_loss={avg_loss:.6f}  pos_steps={epoch_num}\n"
                )