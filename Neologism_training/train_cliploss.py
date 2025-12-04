import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# NEW: CLIP + torchvision for image preprocessing
from transformers import CLIPModel
import torchvision.transforms.functional as TF

from qwen_diff_train_forward import qwen_edit_forward  # your custom forward


# ===========================
# 0) Config (A100 40GB)
# ===========================
load_dotenv()
DATASET_PATH = "/content/sample_data/filtered_dataset.csv"

MODEL_PATH = "Qwen/Qwen-Image-Edit"  # non-4bit official base

TARGET_WORDS = [" and"]

# Learning / schedule
LR = 3e-4
EPOCHS = 20
STEPS_PER_IMAGE = 1   # safe on A100; bump to 2–3 if you want

SAVE_EVERY_N = 1

# Resolution
HEIGHT = 512
WIDTH  = 512

# Denoising
NUM_STEPS_TRAIN = 8   # training steps for qwen_edit_forward
NUM_STEPS_EVAL  = 15  # nicer previews (no grads)
GUIDANCE = 1.0        # no CFG during training (true_cfg_scale)

MAX_TRAIN_IMAGES = 100  # set int for debug

# Reject loss
USE_REJECT = False
REJECT_LAMBDA = 0.00

SAVE_DIR = "./Neologism_training/outputs_neologism_and"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ✅ A100 supports bf16; use it always for Qwen pipeline
DTYPE = torch.bfloat16
print("Using dtype:", DTYPE)


# ===========================
# 1) Load pipeline (frozen)
# ===========================
pipeline = QwenImageEditPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=None,
).to(device)

# Memory helpers (lightweight, safe)
if hasattr(pipeline, "enable_attention_slicing"):
    pipeline.enable_attention_slicing("max")
if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
    pipeline.vae.enable_slicing()
if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
    pipeline.vae.enable_tiling()

# ✅ Make checkpointing actually apply to denoiser
if hasattr(pipeline, "transformer") and hasattr(pipeline.transformer, "enable_gradient_checkpointing"):
    pipeline.transformer.enable_gradient_checkpointing()
else:
    try:
        pipeline.transformer.gradient_checkpointing = True
    except Exception:
        pass

# ✅ Efficient attention (SDPA by default on torch>=2)
try:
    if hasattr(pipeline.transformer, "set_attn_processor"):
        pipeline.transformer.set_attn_processor("sdpa")
except Exception:
    pass

# ✅ xFormers if installed (optional)
try:
    if hasattr(pipeline.transformer, "enable_xformers_memory_efficient_attention"):
        pipeline.transformer.enable_xformers_memory_efficient_attention()
except Exception:
    pass


tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
vae = pipeline.vae

# Freeze everything except our neologism embedding
for module_name in ["text_encoder", "transformer", "vae", "image_encoder"]:
    if hasattr(pipeline, module_name):
        for p in getattr(pipeline, module_name).parameters():
            p.requires_grad_(False)


# ===========================
# 1.5) CLIP model for loss
# ===========================
# We'll use CLIP image embeddings and cosine similarity.
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad_(False)  # we don't train CLIP

# CLIP normalization constants (standard OpenAI CLIP)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

def clip_encode_from_tensor(img_tensor: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
    """
    img_tensor: [B, 3, H, W] in [0,1]
    Returns L2-normalized CLIP image embeddings [B, D].
    """
    # Resize to CLIP resolution
    img = torch.nn.functional.interpolate(
        img_tensor, size=(224, 224), mode="bicubic", align_corners=False
    )
    img = (img - CLIP_MEAN) / CLIP_STD  # normalize

    if no_grad:
        with torch.no_grad():
            emb = clip_model.get_image_features(pixel_values=img)
    else:
        emb = clip_model.get_image_features(pixel_values=img)

    emb = F.normalize(emb, dim=-1)
    return emb



def clip_encode_from_pil(img: Image.Image) -> torch.Tensor:
    """
    Convenience wrapper for target/reject images (no grad needed).
    """
    t = TF.to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]
    return clip_encode_from_tensor(t, no_grad=True)


def latents_to_image_tensor(latents: torch.Tensor) -> torch.Tensor:
    """
    Convert Qwen 'latent' output (B, C, H, W) from qwen_edit_forward
    into an image tensor [B, 3, H, W] in [0,1] for CLIP loss.
    """
    # Qwen VAE expects (B, C, F, H, W) where F is frames (we use 1)
    latents = latents.to(vae.dtype).unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)

    # Decode to pixels in [-1, 1]
    decoded = vae.decode(latents, return_dict=False)[0][:, :, 0]  # (B, 3, H, W)

    # Map [-1, 1] -> [0, 1]
    img = (decoded / 2 + 0.5).clamp(0, 1)
    return img



# ===========================
# 2) Build neologism embedding for "and"
# ===========================
emb_layer = text_encoder.get_input_embeddings()
emb_dtype = emb_layer.weight.dtype
print(f"Text embedding dim: {emb_layer.weight.shape[-1]}, dtype: {emb_dtype}")

target_token_ids: List[int] = []
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

# ✅ Learn neologism in fp32 for stability, rest of model bf16
and_neologism_emb = nn.Parameter(init_vec.float().clone().detach()).to(device)
and_neologism_emb.requires_grad_(True)

opt = torch.optim.AdamW([and_neologism_emb], lr=LR)


# ===========================
# 3) Utilities
# ===========================
def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def parse_img_field(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unexpected type for image field: {type(value)}")


def save_neologism_checkpoint(epoch_idx: int):
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


def neologism_distance_stats():
    cur = and_neologism_emb.detach()
    orig = orig_and_emb
    l2_dist = torch.norm(cur - orig).item()
    cosine_sim = F.cosine_similarity(cur.unsqueeze(0), orig.unsqueeze(0)).item()
    return l2_dist, cosine_sim


# ===========================
# 4) Training loop (CLIP + reject loss)
# ===========================
def train_on_df(df: pd.DataFrame, epochs: int = EPOCHS, steps_per_image: int = STEPS_PER_IMAGE):
    n_rows = len(df)

    # One-time tokenizer sanity check
    first_prompt = str(df.iloc[0]["instruction"])
    enc = tokenizer(first_prompt, add_special_tokens=True)
    print("=== TOKENIZER SANITY CHECK ===")
    print("prompt:", first_prompt)
    print("ids:", enc["input_ids"])
    print("tokens:", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
    print("target_token_ids:", target_token_ids)
    print("================================")

    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")
        epoch_loss_pos_sum = 0.0
        epoch_loss_neg_sum = 0.0
        epoch_num_pos = 0
        epoch_num_neg = 0

        for idx, row in df.iterrows():
            loss_pos = None
            loss_neg = None

            try:
                src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
                tgt_img = pil_from_bytes(parse_img_field(row["target_img"])["bytes"])
                prompt  = str(row["instruction"])
                print(f"Prompt is: {prompt}")

                # Precompute target CLIP embedding (no grad needed)
                tgt_clip = clip_encode_from_pil(tgt_img)

                reject_clip = None
                if USE_REJECT and "reject_img" in df.columns and not pd.isna(row["reject_img"]):
                    rej_img = pil_from_bytes(parse_img_field(row["reject_img"])["bytes"])
                    reject_clip = clip_encode_from_pil(rej_img)

                for step in range(steps_per_image):
                    opt.zero_grad(set_to_none=True)

                    # 1) Generate latents with qwen_edit_forward (autocast bf16)
                    with torch.autocast("cuda", dtype=DTYPE):
                        out = qwen_edit_forward(
                            pipeline,
                            image=src_img,
                            prompt=prompt,
                            negative_prompt=None,
                            num_inference_steps=NUM_STEPS_TRAIN,
                            true_cfg_scale=GUIDANCE,
                            height=HEIGHT,
                            width=WIDTH,
                            output_type="latent",
                            return_dict=True,
                            and_neologism_emb=and_neologism_emb,
                            target_token_ids=target_token_ids,
                        )
                        gen_lat = out.images  # latents

                    if not torch.isfinite(gen_lat).all():
                        print(f"[WARN] Row {idx}, step {step}: non-finite gen_lat; skipping.")
                        continue

                    # 2) Decode latents to image tensor (for CLIP)
                    gen_img = latents_to_image_tensor(gen_lat)  # [B,3,H,W] in [0,1]

                    # 3) CLIP encode generated image WITH grad
                    gen_clip = clip_encode_from_tensor(gen_img, no_grad=False)  # [B,D]

                    # 4) CLIP cosine losses (positive + reject)
                    loss_pos = 1.0 - F.cosine_similarity(gen_clip, tgt_clip, dim=-1).mean()

                    if USE_REJECT and reject_clip is not None:
                        loss_neg = 1.0 - F.cosine_similarity(gen_clip, reject_clip, dim=-1).mean()
                        loss = loss_pos - REJECT_LAMBDA * loss_neg
                    else:
                        loss = loss_pos

                    if not torch.isfinite(loss):
                        print(f"[WARN] Row {idx}, step {step}: non-finite loss; skipping.")
                        continue

                    # LOGGING (per-step accumulation)
                    epoch_loss_pos_sum += loss_pos.item()
                    epoch_num_pos += 1
                    if USE_REJECT and reject_clip is not None and loss_neg is not None:
                        epoch_loss_neg_sum += loss_neg.item()
                        epoch_num_neg += 1

                    # Backprop only w.r.t. and_neologism_emb
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([and_neologism_emb], 1.0)
                    opt.step()

                if (idx + 1) % 5 == 0 and loss_pos is not None:
                    msg = f"[{idx+1}/{n_rows}] loss_pos={loss_pos.item():.4f}"
                    if USE_REJECT and loss_neg is not None:
                        msg += f" loss_neg={loss_neg.item():.4f}"
                    msg += f" ||and_emb||={and_neologism_emb.norm().item():.3f}"
                    print(msg)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Row {idx} failed during training: {e}")

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
            save_neologism_checkpoint(epoch)
            l2_dist, cos_sim = neologism_distance_stats()
            print(f"[CHECKPOINT EPOCH {epoch+1}] L2={l2_dist:.6f} | cosine={cos_sim:.6f}")

            with open(os.path.join(SAVE_DIR, "neologism_drift_log.txt"), "a") as f:
                f.write(f"epoch={epoch+1:03d}  L2={l2_dist:.6f}  cosine={cos_sim:.6f}\n")

            with open(os.path.join(SAVE_DIR, "epoch_loss_log.txt"), "a") as f:
                f.write(
                    f"epoch={epoch+1:03d}  avg_pos_loss={avg_pos:.6f}  "
                    f"avg_neg_loss={avg_neg:.6f}  pos_steps={epoch_num_pos}  "
                    f"neg_steps={epoch_num_neg}\n"
                )


# ===========================
# 5) Evaluation pass (save outputs)
# ===========================
@torch.no_grad()
def evaluate_and_save(df: pd.DataFrame, subdir: str = "eval"):
    out_dir = os.path.join(SAVE_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in df.iterrows():
        try:
            src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
            prompt  = str(row["instruction"])

            # (A) BASELINE: no neologism injection
            out_base = qwen_edit_forward(
                pipeline,
                image=src_img,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=NUM_STEPS_EVAL,
                true_cfg_scale=4.0,     # CFG ok at eval
                height=HEIGHT,
                width=WIDTH,
                output_type="pil",
                return_dict=True,
                and_neologism_emb=None,   # <-- baseline
                target_token_ids=None,
            )
            img_base = out_base.images[0]

            # (B) NEOLOGISM: inject trained embedding
            out_neo = qwen_edit_forward(
                pipeline,
                image=src_img,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=NUM_STEPS_EVAL,
                true_cfg_scale=4.0,
                height=HEIGHT,
                width=WIDTH,
                output_type="pil",
                return_dict=True,
                and_neologism_emb=and_neologism_emb,  # <-- neo
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


# ===========================
# 5.5) Initial CLIP loss (epoch 0)
# ===========================
@torch.no_grad()
def compute_initial_epoch_loss(df):
    pos_sum = 0.0
    neg_sum = 0.0
    n_pos = 0
    n_neg = 0

    print("\n=== COMPUTING INITIAL BASELINE CLIP LOSS (epoch 0) ===")

    for idx, row in df.iterrows():
        try:
            src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
            tgt_img = pil_from_bytes(parse_img_field(row["target_img"])["bytes"])

            tgt_clip = clip_encode_from_pil(tgt_img)
            reject_clip = None
            if USE_REJECT and "reject_img" in df.columns and not pd.isna(row["reject_img"]):
                rej_img = pil_from_bytes(parse_img_field(row["reject_img"])["bytes"])
                reject_clip = clip_encode_from_pil(rej_img)

            with torch.autocast("cuda", dtype=DTYPE):
                out = qwen_edit_forward(
                    pipeline,
                    image=src_img,
                    prompt=str(row["instruction"]),
                    negative_prompt=None,
                    num_inference_steps=NUM_STEPS_TRAIN,
                    true_cfg_scale=GUIDANCE,
                    height=HEIGHT,
                    width=WIDTH,
                    output_type="latent",
                    return_dict=True,
                    and_neologism_emb=and_neologism_emb,   # initial embedding
                    target_token_ids=target_token_ids,
                )
            gen_lat = out.images
            gen_img = latents_to_image_tensor(gen_lat)
            gen_clip = clip_encode_from_tensor(gen_img, no_grad=True)

            loss_pos = 1.0 - F.cosine_similarity(gen_clip, tgt_clip, dim=-1).mean()
            pos_sum += loss_pos.item()
            n_pos += 1

            if USE_REJECT and reject_clip is not None:
                loss_neg = 1.0 - F.cosine_similarity(gen_clip, reject_clip, dim=-1).mean()
                neg_sum += loss_neg.item()
                n_neg += 1

        except Exception as e:
            print(f"[WARN] initial loss failed on row {idx}: {e}")

    avg_pos = pos_sum / max(n_pos, 1)
    avg_neg = neg_sum / max(n_neg, 1)

    print(f"Initial avg_pos_loss={avg_pos:.6f} over {n_pos} samples")
    if USE_REJECT:
        print(f"Initial avg_neg_loss={avg_neg:.6f} over {n_neg} samples")
    print("=====================================\n")

    return avg_pos, avg_neg


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

    print("Starting pre train eval")
    initial_pos, initial_neg = compute_initial_epoch_loss(df)
    with open(os.path.join(SAVE_DIR, "epoch_loss_log.txt"), "a") as f:
        f.write(
            f"epoch=000  avg_pos_loss={initial_pos:.6f}  "
            f"avg_neg_loss={initial_neg:.6f}\n"
        )

    print("Initial neologism norm:", and_neologism_emb.norm().item())
    print("Starting training...")
    train_on_df(df)
    print("Final neologism norm:", and_neologism_emb.norm().item())

    print("Evaluating...")
    evaluate_and_save(df, subdir="eval")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Bake learned embedding back into embedding table
    with torch.no_grad():
        new_vec = and_neologism_emb.to(emb_layer.weight.dtype).to(emb_layer.weight.device)
        for tid in target_token_ids:
            emb_layer.weight[tid] = new_vec

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

    print("Done. Outputs in:", SAVE_DIR)
