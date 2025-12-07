import torch
import gc

def generate_with_gradients(pipe, image, prompt, num_inference_steps=8, guidance_scale=7.5, image_guidance_scale=1.5, neologism_emb=None, target_token_ids=None):
    """
    Generate latents with gradient flow through the InstructPix2Pix pipeline.
    Modified https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
    """
    pipe._guidance_scale = guidance_scale
    pipe._image_guidance_scale = image_guidance_scale
    device = pipe._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # get prompt embeddings (w/o neologism)
    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        1, # num_images_per_prompt
        pipe.do_classifier_free_guidance,
    )

    # Preprocess image
    image = pipe.image_processor.preprocess(image)

    # inject neologism embedding
    if neologism_emb is not None and target_token_ids is not None and prompt is not None:
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

    # Tokenize pure text to know where "and" is in token space
    encoded = pipe.tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(prompt_embeds.device)  # (B_text, T_text)

    B_text, T_text = input_ids.shape
    B_emb, L, D = prompt_embeds.shape  # full sequence from encode_prompt

    # Heuristic: assume last T_text tokens in prompt_embeds are text tokens
    if T_text <= L:
        text_start = L - T_text
    else:
        text_start = 0
        T_text = min(T_text, L)

    tok_ids_set = set(int(t) for t in target_token_ids)

    # Build mask over sequence positions where token is "and"
    mask = torch.zeros((B_emb, L), dtype=torch.bool, device=prompt_embeds.device)
    for b in range(B_text):
        for t in range(T_text):
            if int(input_ids[b, t]) in tok_ids_set:
                mask[b, text_start + t] = True

    # Convert boolean mask to float for arithmetic
    mask_f = mask.unsqueeze(-1).to(prompt_embeds.dtype)  # (B, L, 1)

    neo_vec = neologism_emb.view(1, 1, -1).to(prompt_embeds.dtype).to(prompt_embeds.device)

    # New embeddings:
    #   E_new = E_base * (1 - mask) + neo_vec * mask
    # This *always* reassigns prompt_embeds, so it always depends on neo_vec.
    prompt_embeds = prompt_embeds * (1.0 - mask_f) + neo_vec * mask_f

    del mask, mask_f, input_ids, encoded
    torch.cuda.empty_cache()

    # set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # prepare image latents
    image_latents = pipe.prepare_image_latents(
        image,
        batch_size,
        1, # num_images_per_prompt
        prompt_embeds.dtype,
        device,
        pipe.do_classifier_free_guidance,
    )

    height, width = image_latents.shape[-2:]
    height = height * pipe.vae_scale_factor
    width = width * pipe.vae_scale_factor

    # prepare latent variables
    num_channels_latents = pipe.vae.config.latent_channels
    latents = pipe.prepare_latents(
        batch_size * 1, 
        num_channels_latents, # batch_size * num_images_per_prompt
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator=None,
        latents=None
    )

    # check that shapes of latents and image match the UNet channels
    num_channels_image = image_latents.shape[1]
    if num_channels_latents + num_channels_image != pipe.unet.config.in_channels:
        raise ValueError(
            f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
            f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
            f" `num_channels_image`: {num_channels_image} "
            f" = {num_channels_latents + num_channels_image}. Please verify the config of"
            " `pipeline.unet` or your `image` input."
        )

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

    # denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance.
            # The latents are expanded 3 times because for pix2pix the guidance\
            # is applied for both the text and the input image.
            latent_model_input = torch.cat([latents] * 3) if pipe.do_classifier_free_guidance else latents

            # concat latents, image_latents in the channel dimension
            scaled_latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

            # predict the noise residual
            noise_pred = pipe.unet(
                scaled_latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=None,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + pipe.guidance_scale * (noise_pred_text - noise_pred_image)
                    + pipe.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

    return latents

def inject_neologism_embedding(pipe, prompt, prompt_embeds, neologism_emb, target_token_ids):
    # inject neologism embedding
    if neologism_emb is not None and target_token_ids is not None and prompt is not None:
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

    # Tokenize pure text to know where "and" is in token space
    encoded = pipe.tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(prompt_embeds.device)  # (B_text, T_text)

    B_text, T_text = input_ids.shape
    B_emb, L, D = prompt_embeds.shape  # full sequence from encode_prompt

    # Heuristic: assume last T_text tokens in prompt_embeds are text tokens
    if T_text <= L:
        text_start = L - T_text
    else:
        text_start = 0
        T_text = min(T_text, L)

    tok_ids_set = set(int(t) for t in target_token_ids)

    # Build mask over sequence positions where token is "and"
    mask = torch.zeros((B_emb, L), dtype=torch.bool, device=prompt_embeds.device)
    for b in range(B_text):
        for t in range(T_text):
            if int(input_ids[b, t]) in tok_ids_set:
                mask[b, text_start + t] = True

    # Convert boolean mask to float for arithmetic
    mask_f = mask.unsqueeze(-1).to(prompt_embeds.dtype)  # (B, L, 1)

    neo_vec = neologism_emb.view(1, 1, -1).to(prompt_embeds.dtype).to(prompt_embeds.device)

    # New embeddings:
    #   E_new = E_base * (1 - mask) + neo_vec * mask
    # This *always* reassigns prompt_embeds, so it always depends on neo_vec.
    prompt_embeds = prompt_embeds * (1.0 - mask_f) + neo_vec * mask_f

    del mask, mask_f, input_ids, encoded
    torch.cuda.empty_cache()

    return prompt_embeds

def generate_with_gradients_lang(pipe, prompt, neologism_emb=None, target_token_ids=None):
    """
    Generate latents with gradient flow through the InstructPix2Pix pipeline.
    Modified https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
    """
    pipe._guidance_scale = 7.5
    pipe._image_guidance_scale = 1.0
    device = pipe._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    # get prompt embeddings (w/o neologism)
    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        1, # num_images_per_prompt
        pipe.do_classifier_free_guidance,
    )
    prompt_embeds = inject_neologism_embedding(pipe, prompt, prompt_embeds, neologism_emb, target_token_ids)

    return prompt_embeds  

def decode_latents_to_image(pipe, latents):
    """
    Decode latents to PIL image.
    
    Args:
        pipe: StableDiffusionInstructPix2PixPipeline
        latents: torch.Tensor [B, 4, H//8, W//8]
    
    Returns:
        PIL.Image: Decoded image
    """
    # Decode latents
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    
    # Post-process
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])
    
    return image


def latents_to_tensor(pipe, latents):
    """
    Decode latents to image tensor [B, 3, H, W] in range [0, 1] with gradients.
    This is what you want for CLIP encoding during training.
    
    Args:
        pipe: StableDiffusionInstructPix2PixPipeline
        latents: torch.Tensor [B, 4, H//8, W//8]
    
    Returns:
        torch.Tensor: Image [B, 3, H, W] in [0, 1] with gradients
    """

    with torch.set_grad_enabled(True):
        # Decode latents (this maintains gradients)
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        
        # Convert from [-1, 1] to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
    
    return image

    