"""
Eval neologism embedding for AND
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
from instruct_pix2pix_diff_forward import generate_with_gradients, decode_latents_to_image
from utils.logging_utils import update_csv
from PIL import Image
import time


TARGET_WORDS = [" and"]
STEPS_PER_IMAGE = 1
MAX_TRAIN_IMAGES = 80
EPOCHS = 100
NUM_STEPS_TRAIN=8
SAVE_DIR = f"./instruct_pix2pix/results/instruct_pix2pix_outputs_neologism_and_{STEPS_PER_IMAGE}stepsPerImage_{MAX_TRAIN_IMAGES}trainImages_{EPOCHS}epochs_{NUM_STEPS_TRAIN}denoisingSteps"

# the and neologism embedding weights to use
AND_NEOLOGISM_EPOCH = 100

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

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

    # Freeze everything
    modules = ["text_encoder", "vae", "unet"]
    for module_name in modules:
        if hasattr(pipe, module_name):
            for p in getattr(pipe, module_name).parameters():
                p.requires_grad_(False)

    # Build neologism embedding for "and"
    emb_layer = text_encoder.get_input_embeddings()
    emb_dtype = emb_layer.weight.dtype
    print(f"Text embedding dim: {emb_layer.weight.shape[-1]}, dtype: {emb_dtype}")

    # get the embedding for "and"
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


    ckpt_path = f"{SAVE_DIR}/and_neologism_epoch_{AND_NEOLOGISM_EPOCH:03d}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    and_neologism_emb = ckpt["embedding"]

    path_to_images = '../../scratch/DL_data/images'
    path_to_prompt = '../../scratch/DL_data/prompts/composite'

    input_path = f'{path_to_images}/original'
    output_dir = f'{path_to_images}/instruct_pix2pix_outputs_neologism_and_{STEPS_PER_IMAGE}stepsPerImage_{MAX_TRAIN_IMAGES}trainImages_{EPOCHS}epochs_{NUM_STEPS_TRAIN}denoisingSteps_ckpt{AND_NEOLOGISM_EPOCH}'
    os.makedirs(output_dir, exist_ok=True)

    csv_path = f'{output_dir}/times.csv'

    for fname in os.listdir(input_path):
        start_time = time.time()

        file_name = fname[:-4] # removes ".png"

        input_image = Image.open(f"{input_path}/{file_name}.png").convert('RGB')
        prompt = open(f"{path_to_prompt}/{file_name}.txt", 'r').read()

        torch.cuda.empty_cache()

        with torch.inference_mode():
            image_latents = generate_with_gradients(
                pipe,
                input_image,
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                image_guidance_scale=1.5,
                neologism_emb=and_neologism_emb,
                target_token_ids=target_token_ids
            )
            image = decode_latents_to_image(pipe, image_latents)[0]
            image.save(f"{output_dir}/{file_name}.png")

        torch.cuda.empty_cache()

        time_taken = time.time() - start_time
        update_csv(csv_path, file_name, time_taken)
        print(f"Edited image saved to {output_dir}/{file_name}.png | Time: {time_taken}s")