import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import gc
import ast
from typing import List, Tuple, Optional
from dotenv import load_dotenv

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImageEditPipeline


from transformers import CLIPModel
import torchvision.transforms.functional as TF

from qwen_diff_train_forward import qwen_edit_forward  # custom forward for training



load_dotenv()
DATASET_PATH = "/content/sample_data/filtered_dataset.csv"

MODEL_PATH = "Qwen/Qwen-Image-Edit"  

TARGET_WORDS = [" and"]

LR = 3e-4
EPOCHS = 20
STEPS_PER_IMAGE = 1   

SAVE_EVERY_N = 1


HEIGHT = 512
WIDTH  = 512

# Denoising
NUM_STEPS_TRAIN = 8   
NUM_STEPS_EVAL  = 15  
GUIDANCE = 1.0        # no CFG during training 

MAX_TRAIN_IMAGES = 100 


USE_REJECT = False
REJECT_LAMBDA = 0.00


MARGIN = 0.02       
LAMBDA_ABS = 0.5  


W_AB = 0.5
W_A  = 0.25
W_B  = 0.25

SAVE_DIR = "./Neologism_training/outputs_neologism_and"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


DTYPE = torch.bfloat16
print("Using dtype:", DTYPE)


pipeline = QwenImageEditPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=None,
).to(device)

# Memory helpers (lightweight, safe) (source: from chatGPT)
if hasattr(pipeline, "enable_attention_slicing"):
    pipeline.enable_attention_slicing("max")
if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
    pipeline.vae.enable_slicing()
if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
    pipeline.vae.enable_tiling()

if hasattr(pipeline, "transformer") and hasattr(pipeline.transformer, "enable_gradient_checkpointing"):
    pipeline.transformer.enable_gradient_checkpointing()
else:
    try:
        pipeline.transformer.gradient_checkpointing = True
    except Exception:
        pass

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


tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
vae = pipeline.vae

for module_name in ["text_encoder", "transformer", "vae", "image_encoder"]:
    if hasattr(pipeline, module_name):
        for p in getattr(pipeline, module_name).parameters():
            p.requires_grad_(False)



CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad_(False) 


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

def clip_encode_from_tensor(img_tensor: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
    """
    img_tensor: [B, 3, H, W] in [0,1]
    Returns L2-normalized CLIP image embeddings [B, D].
    """
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
    t = TF.to_tensor(img).unsqueeze(0).to(device)  
    return clip_encode_from_tensor(t, no_grad=True)


def latents_to_image_tensor(latents: torch.Tensor) -> torch.Tensor:
    """
    Convert Qwen 'latent' output (B, C, H, W) from qwen_edit_forward
    into an image tensor [B, 3, H, W] in [0,1] for CLIP loss.
    """
    latents = latents.to(vae.dtype).unsqueeze(2)  
    decoded = vae.decode(latents, return_dict=False)[0][:, :, 0] 
    img = (decoded / 2 + 0.5).clamp(0, 1)  
    return img



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

# Initialize neologism to a semantically vacuous vector (vocab mean)
with torch.no_grad():
    and_vecs = emb_layer.weight[target_token_ids].to(device=device, dtype=emb_dtype)
    orig_and_semantic = and_vecs.mean(dim=0).clone()  

    # Semantically vacuous vector
    vocab_mean = emb_layer.weight.mean(dim=0).to(device=device, dtype=emb_dtype)

    #neologism start (vacuous)
    orig_and_emb = vocab_mean.float().detach().clone().to(device)

    print(
        f"Initialized neologism from GLOBAL vocab mean "
        f"(semantically vacuous), target_token_ids={target_token_ids}"
    )


and_neologism_emb = nn.Parameter(orig_and_emb.float().clone().detach()).to(device)
and_neologism_emb.requires_grad_(True)

opt = torch.optim.AdamW([and_neologism_emb], lr=LR)


with torch.no_grad():
    vocab_norm_mean = emb_layer.weight.norm(dim=1).mean().item()


vocab_mean_fp32 = vocab_mean.float().detach().clone().to(device)

# Regularization hyperparams
LAMBDA_NORM   = 0.10  
LAMBDA_MEAN   = 0.05  
LAMBDA_ORTHO  = 0.02   



def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def parse_img_field(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unexpected type for image field: {type(value)}")


def save_neologism_checkpoint(epoch_idx: int):
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
    """
    Drift relative to the *vacuous* initialization (vocab mean).
    If you also want drift from true 'and', you can expand this later.
    """
    cur = and_neologism_emb.detach()
    vac = orig_and_emb
    l2_dist = torch.norm(cur - vac).item()
    cosine_sim = F.cosine_similarity(cur.unsqueeze(0), vac.unsqueeze(0)).item()
    return l2_dist, cosine_sim



def split_instruction_on_and(instr: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Very simple heuristic: split on first ' and '.
    Returns (taskA, taskB) or (None, None) if no clean split.
    """
    marker = " and "
    if marker not in instr:
        return None, None
    parts = instr.split(marker, 1)
    taskA = parts[0].strip()
    taskB = parts[1].strip()
    if not taskA or not taskB:
        return None, None
    return taskA, taskB



@torch.no_grad()
def precompute_target_and_baseline(df: pd.DataFrame):
    n_rows = len(df)
    tgt_clips = [None] * n_rows
    baseline_sims = [None] * n_rows

    print("\n=== PRECOMPUTING target CLIP and baseline similarities ===")
    for idx, row in df.iterrows():
        try:
            src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
            tgt_img = pil_from_bytes(parse_img_field(row["target_img"])["bytes"])

            tgt_clip = clip_encode_from_pil(tgt_img)  
            tgt_clips[idx] = tgt_clip.squeeze(0).cpu()

            with torch.autocast("cuda", dtype=DTYPE):
                out_base = qwen_edit_forward(
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
                    and_neologism_emb=None,  
                    target_token_ids=None,
                )
            lat_base = out_base.images
            img_base = latents_to_image_tensor(lat_base)
            base_clip = clip_encode_from_tensor(img_base, no_grad=True)

            sim_base = F.cosine_similarity(base_clip, tgt_clip, dim=-1).mean().item()
            baseline_sims[idx] = sim_base

            if (idx + 1) % 10 == 0:
                print(f"[PRECOMPUTE] {idx+1}/{n_rows} rows done")

        except Exception as e:
            print(f"[WARN] Precompute failed on row {idx}: {e}")

    print("=== PRECOMPUTE DONE ===\n")
    return tgt_clips, baseline_sims


@torch.no_grad()
def precompute_single_task_targets(df: pd.DataFrame):
    """
    For each row with an 'A and B' instruction, generate:
      - A-only edit from the current baseline pipeline
      - B-only edit from the current baseline pipeline
    and store their CLIP embeddings as pseudo-targets.

    Returns:
        tgtA_clips, tgtB_clips: lists of [D] tensors or None.
    """
    n_rows = len(df)
    tgtA_clips = [None] * n_rows
    tgtB_clips = [None] * n_rows

    print("\n=== PRECOMPUTING pseudo Task-A and Task-B CLIP embeddings ===")
    for idx, row in df.iterrows():
        try:
            instr = str(row["instruction"])
            taskA, taskB = split_instruction_on_and(instr)
            if taskA is None:
                continue

            src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])

            prompt_A = (
                f"{taskA}. Only apply this change; do NOT {taskB.lower()}."
            )
            with torch.autocast("cuda", dtype=DTYPE):
                outA = qwen_edit_forward(
                    pipeline,
                    image=src_img,
                    prompt=prompt_A,
                    negative_prompt=None,
                    num_inference_steps=NUM_STEPS_TRAIN,
                    true_cfg_scale=GUIDANCE,
                    height=HEIGHT,
                    width=WIDTH,
                    output_type="pil",
                    return_dict=True,
                    and_neologism_emb=None,   
                    target_token_ids=None,
                )
            imgA = outA.images[0]
            clipA = clip_encode_from_pil(imgA)  
            tgtA_clips[idx] = clipA.squeeze(0).cpu()

            prompt_B = (
                f"{taskB}. Only apply this change; do NOT {taskA.lower()}."
            )
            with torch.autocast("cuda", dtype=DTYPE):
                outB = qwen_edit_forward(
                    pipeline,
                    image=src_img,
                    prompt=prompt_B,
                    negative_prompt=None,
                    num_inference_steps=NUM_STEPS_TRAIN,
                    true_cfg_scale=GUIDANCE,
                    height=HEIGHT,
                    width=WIDTH,
                    output_type="pil",
                    return_dict=True,
                    and_neologism_emb=None,  
                    target_token_ids=None,
                )
            imgB = outB.images[0]
            clipB = clip_encode_from_pil(imgB)
            tgtB_clips[idx] = clipB.squeeze(0).cpu()

            if (idx + 1) % 10 == 0:
                print(f"[PRECOMPUTE A/B] {idx+1}/{n_rows} rows done")

        except Exception as e:
            print(f"[WARN] precompute_single_task_targets failed on row {idx}: {e}")

    print("=== PRECOMPUTE A/B DONE ===\n")
    return tgtA_clips, tgtB_clips


def train_on_df(
    df: pd.DataFrame,
    tgt_clips,
    baseline_sims,
    tgtA_clips,
    tgtB_clips,
    epochs: int = EPOCHS,
    steps_per_image: int = STEPS_PER_IMAGE,
):
    n_rows = len(df)

    # tokenizer sanity check
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
        epoch_loss_abs_sum   = 0.0
        epoch_loss_rank_sum  = 0.0
        epoch_loss_norm_sum  = 0.0
        epoch_loss_mean_sum  = 0.0
        epoch_loss_orth_sum  = 0.0
        epoch_num_steps      = 0

        for idx, row in df.iterrows():
            try:
                # Skip rows where precompute failed
                if tgt_clips[idx] is None or baseline_sims[idx] is None:
                    print(f"[SKIP] Row {idx} missing precomputed AB or baseline.")
                    continue

                src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
                prompt  = str(row["instruction"])
                print(f"Prompt is: {prompt}")

                tgt_clip = tgt_clips[idx].unsqueeze(0).to(device)  
                sim_base_val = baseline_sims[idx]  

                tgtA_clip = None
                tgtB_clip = None
                if tgtA_clips is not None and tgtA_clips[idx] is not None:
                    tgtA_clip = tgtA_clips[idx].unsqueeze(0).to(device)
                if tgtB_clips is not None and tgtB_clips[idx] is not None:
                    tgtB_clip = tgtB_clips[idx].unsqueeze(0).to(device)

                for step in range(steps_per_image):
                    opt.zero_grad(set_to_none=True)

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
                        gen_lat = out.images  

                    if not torch.isfinite(gen_lat).all():
                        print(f"[WARN] Row {idx}, step {step}: non-finite gen_lat; skipping.")
                        continue

                    gen_img = latents_to_image_tensor(gen_lat)  

                    gen_clip = clip_encode_from_tensor(gen_img, no_grad=False)  


                    sim_AB = F.cosine_similarity(gen_clip, tgt_clip, dim=-1).mean()
                    loss_AB = 1.0 - sim_AB

                    sim_base = torch.tensor(
                        sim_base_val,
                        device=device,
                        dtype=sim_AB.dtype,
                    )
                    loss_rank = F.relu(MARGIN - sim_AB + sim_base)

                    loss_A = torch.tensor(0.0, device=device, dtype=sim_AB.dtype)
                    loss_B = torch.tensor(0.0, device=device, dtype=sim_AB.dtype)

                    if tgtA_clip is not None:
                        sim_A = F.cosine_similarity(gen_clip, tgtA_clip, dim=-1).mean()
                        loss_A = 1.0 - sim_A

                    if tgtB_clip is not None:
                        sim_B = F.cosine_similarity(gen_clip, tgtB_clip, dim=-1).mean()
                        loss_B = 1.0 - sim_B

                    # Loss of A and B computed as well so that it doesnt maximize AB by maximizing either a or b
                    loss_abs = W_AB * loss_AB + W_A * loss_A + W_B * loss_B

                    # Final task loss == absolute + ranking
                    loss_task = LAMBDA_ABS * loss_abs + (1.0 - LAMBDA_ABS) * loss_rank


                    e = and_neologism_emb

                    # Norm hinge - keep ||e|| close to typical vocab norm (kinda redundant but works well in practice???)
                    norm_e = e.norm()
                    loss_norm = LAMBDA_NORM * F.relu(norm_e - vocab_norm_mean)

                    # Mean attraction - stay near mean of entrie vocab
                    loss_mean = LAMBDA_MEAN * (e - vocab_mean_fp32).pow(2).mean()

                    #Orthogonal drift - don't stray too far from init
                    loss_orth = LAMBDA_ORTHO * (e - orig_and_emb).pow(2).mean()

                    # Full loss
                    loss = loss_task + loss_norm + loss_mean + loss_orth

                    if not torch.isfinite(loss):
                        print(f"[WARN] Row {idx}, step {step}: non-finite loss; skipping.")
                        continue

                
                    epoch_loss_abs_sum  += loss_abs.item()
                    epoch_loss_rank_sum += loss_rank.item()
                    epoch_loss_norm_sum += loss_norm.item()
                    epoch_loss_mean_sum += loss_mean.item()
                    epoch_loss_orth_sum += loss_orth.item()
                    epoch_num_steps     += 1

                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([and_neologism_emb], 1.0)
                    opt.step()

                if (idx + 1) % 5 == 0:
                    msg = (
                        f"[{idx+1}/{n_rows}] "
                        f"loss_abs={loss_abs.item():.4f} "
                        f"loss_rank={loss_rank.item():.4f} "
                        f"||and_emb||={and_neologism_emb.norm().item():.3f}"
                    )
                    print(msg)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Row {idx} failed during training: {e}")

        # Epoch-wise averages
        if epoch_num_steps > 0:
            avg_abs   = epoch_loss_abs_sum  / epoch_num_steps
            avg_rank  = epoch_loss_rank_sum / epoch_num_steps
            avg_norm  = epoch_loss_norm_sum / epoch_num_steps
            avg_mean  = epoch_loss_mean_sum / epoch_num_steps
            avg_orth  = epoch_loss_orth_sum / epoch_num_steps
        else:
            avg_abs = avg_rank = avg_norm = avg_mean = avg_orth = float("nan")

        print(
            f"[EPOCH {epoch+1}] "
            f"avg_abs_loss={avg_abs:.6f}, "
            f"avg_rank_loss={avg_rank:.6f}, "
            f"avg_norm_loss={avg_norm:.6f}, "
            f"avg_mean_loss={avg_mean:.6f}, "
            f"avg_orth_loss={avg_orth:.6f} "
            f"(steps={epoch_num_steps})"
        )

        if (epoch + 1) % SAVE_EVERY_N == 0:
            save_neologism_checkpoint(epoch)
            l2_dist, cos_sim = neologism_distance_stats()
            print(f"[CHECKPOINT EPOCH {epoch+1}] L2={l2_dist:.6f} | cosine={cos_sim:.6f}")

            with open(os.path.join(SAVE_DIR, "neologism_drift_log.txt"), "a") as f:
                f.write(f"epoch={epoch+1:03d}  L2={l2_dist:.6f}  cosine={cos_sim:.6f}\n")

            with open(os.path.join(SAVE_DIR, "epoch_loss_log.txt"), "a") as f:
                f.write(
                    f"epoch={epoch+1:03d}  "
                    f"avg_abs_loss={avg_abs:.6f}  "
                    f"avg_rank_loss={avg_rank:.6f}  "
                    f"avg_norm_loss={avg_norm:.6f}  "
                    f"avg_mean_loss={avg_mean:.6f}  "
                    f"avg_orth_loss={avg_orth:.6f}  "
                    f"steps={epoch_num_steps}\n"
                )



@torch.no_grad()
def evaluate_and_save(df: pd.DataFrame, subdir: str = "eval"):
    out_dir = os.path.join(SAVE_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in df.iterrows():
        try:
            src_img = pil_from_bytes(parse_img_field(row["source_img"])["bytes"])
            prompt  = str(row["instruction"])

            #BASELINE - no neologism injection
            out_base = qwen_edit_forward(
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
                and_neologism_emb=None,   
                target_token_ids=None,
            )
            img_base = out_base.images[0]

            # NEOLOGISM - inject trained embedding
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


#Compute loss on the random initilized embedding
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


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)

    # reject_img is optional now
    required_cols = {"source_img", "target_img", "instruction"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    print(f"Loaded dataset with shape {df.shape}")

    if MAX_TRAIN_IMAGES is not None:
        df = df.head(MAX_TRAIN_IMAGES)
        print(f"Using first {len(df)} rows for training")

    print("Precomputing target CLIP embeddings and baseline similarities...")
    tgt_clips, baseline_sims = precompute_target_and_baseline(df)

    print("Precomputing pseudo Task-A and Task-B CLIP embeddings...")
    tgtA_clips, tgtB_clips = precompute_single_task_targets(df)

    print("Starting pre train eval (initial neologism embedding)...")
    initial_pos, initial_neg = compute_initial_epoch_loss(df)
    with open(os.path.join(SAVE_DIR, "epoch_loss_log.txt"), "a") as f:
        f.write(
            f"epoch=000  avg_pos_loss={initial_pos:.6f}  "
            f"avg_neg_loss={initial_neg:.6f}\n"
        )

    print("Initial neologism norm:", and_neologism_emb.norm().item())
    print("Starting training with multi-target CLIP + ranking + manifold regularization...")
    train_on_df(df, tgt_clips, baseline_sims, tgtA_clips, tgtB_clips)
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
