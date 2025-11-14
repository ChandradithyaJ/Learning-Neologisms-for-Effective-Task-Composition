# qwen_diff_train_forward.py

from typing import Optional, List, Dict, Any, Union, Callable
import inspect
import numpy as np
import torch
import logging

from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage import QwenImagePipelineOutput

logger = logging.getLogger(__name__)


# -------------------------
# Scheduler helpers
# -------------------------

def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """
    Simple linear mapping from image_seq_len to a shift mu.
    This mirrors what the official Qwen / Flux pipelines do when use_dynamic_shifting=True.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return float(mu)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Copied from diffusers' retrieve_timesteps pattern.
    We pass **kwargs through so we can supply mu=... when use_dynamic_shifting=True.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )

    if timesteps is not None:
        accepts_timesteps = "timesteps" in inspect.signature(scheduler.set_timesteps).parameters
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` "
                "does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        accepts_sigmas = "sigmas" in inspect.signature(scheduler.set_timesteps).parameters
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` "
                "does not support custom sigma schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


# -------------------------
# Differentiable forward
# -------------------------

def qwen_edit_forward(
    pipe: QwenImageEditPipeline,
    image=None,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: Optional[float] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    # NEW: neologism injection
    and_neologism_emb: Optional[torch.Tensor] = None,
    target_token_ids: Optional[List[int]] = None,
):
    """
    Differentiable clone of QwenImageEditPipeline.__call__.

    Key differences:
      - No @torch.no_grad().
      - Uses FlowMatch scheduler with dynamic shifting (mu).
      - After encode_prompt, we override "and" token embeddings in prompt_embeds
        with `and_neologism_emb`, so the final images depend on that parameter.
    """

    # ---------- 0. Height / width ----------
    if height is None or width is None:
        if isinstance(image, list):
            w0, h0 = image[0].size
        else:
            w0, h0 = image.size
        height = height or h0
        width = width or w0

    multiple_of = pipe.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    # ---------- 1. Check inputs ----------
    pipe.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    # ---------- 2. Batch size ----------
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    # ---------- 3. Preprocess image ----------
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == pipe.latent_channels):
        image = pipe.image_processor.resize(image, height, width)
        prompt_image = image
        image = pipe.image_processor.preprocess(image, height, width)
        image = image.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
    else:
        prompt_image = image

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    if true_cfg_scale > 1 and not has_neg_prompt:
        logger.warning(
            f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
        )
    elif true_cfg_scale <= 1 and has_neg_prompt:
        logger.warning(
            "negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
        )

    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    # ---------- 4. Encode prompts (may be no-grad internally, that's fine) ----------
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        image=prompt_image,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
            image=prompt_image,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # ---------- 4b. Inject neologism embedding into prompt_embeds ----------
    # IMPORTANT: This happens outside any @torch.no_grad, so it is differentiable
    # w.r.t. and_neologism_emb even if encode_prompt was not.
    # ---------- 4b. Inject neologism embedding into *only* the "and" tokens ----------
    # This happens outside any @torch.no_grad, so it is differentiable w.r.t. and_neologism_emb
    if and_neologism_emb is not None and target_token_ids is not None and prompt is not None:
        # Normalize to list of strings
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

        #FOR TESTING
        #for b in range(input_ids.size(0)):     
        #    ids = input_ids[b].tolist()
        #    toks = pipe.tokenizer.convert_ids_to_tokens(ids)
        #    print(f"\n--- Prompt #{b} tokenization ---")
        #    for tok, tid in zip(toks, ids):
        #        print(f"{tok!r} : {tid}")
        ###TESTING
        B_text, T_text = input_ids.shape

        B_emb, L, D = prompt_embeds.shape  # full sequence from encode_prompt

        # Heuristic: assume last T_text tokens in prompt_embeds are text tokens
        if T_text <= L:
            text_start = L - T_text
        else:
            text_start = 0
            T_text = min(T_text, L)

        tok_ids_set = set(int(t) for t in target_token_ids)

        # Build mask over *sequence positions* where token is "and"
        mask = torch.zeros((B_emb, L), dtype=torch.bool, device=prompt_embeds.device)
        #print(f"Looking for these ids: {tok_ids_set}")
        for b in range(B_text):
            for t in range(T_text):
                #print(f"Token ID{input_ids[b, t]}")
                if int(input_ids[b, t]) in tok_ids_set:
                    mask[b, text_start + t] = True

        # Convert boolean mask to float for arithmetic
        mask_f = mask.unsqueeze(-1).to(prompt_embeds.dtype)  # (B, L, 1)

        neo_vec = and_neologism_emb.view(1, 1, -1).to(prompt_embeds.dtype).to(prompt_embeds.device)

        # New embeddings:
        #   E_new = E_base * (1 - mask) + neo_vec * mask
        # This *always* reassigns prompt_embeds, so it always depends on neo_vec.
        prompt_embeds = prompt_embeds * (1.0 - mask_f) + neo_vec * mask_f

        # Debug: how many "and" positions did we hit?
        print("[DEBUG] num_and_tokens:", int(mask.sum().item()))
        print("[DEBUG] prompt_embeds.requires_grad:", prompt_embeds.requires_grad)
        print("[DEBUG] and_neologism_emb.requires_grad:", and_neologism_emb.requires_grad)


    # ---------- 5. Prepare latents ----------
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents = pipe.prepare_latents(
        image,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    img_shapes = [
        [
            (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2),
            (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2),
        ]
    ] * batch_size

    # ---------- 6. Timesteps with dynamic shifting ----------
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas

    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # ---------- 7. Guidance setup ----------
    if pipe.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not pipe.transformer.config.guidance_embeds and guidance_scale is not None:
        logger.warning(
            f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
        )
        guidance = None
    else:
        guidance = None

    if pipe.attention_kwargs is None:
        pipe._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )

    # ---------- 8. Denoising loop ----------
    pipe.scheduler.set_begin_index(0)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            pipe._current_timestep = t

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # conditional
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=pipe.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                # unconditional
                with pipe.transformer.cache_context("uncond"):
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=pipe.attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    pipe._current_timestep = None

    # ---------- 9. Output ----------

    if output_type == "latent":
        # Convert sequence latents (B, seq, dim) -> VAE latents (B, C, H', W')
        latents_img = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents_img = latents_img.to(pipe.vae.dtype)

        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_img.device, latents_img.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents_img.device, latents_img.dtype)

        latents_img = latents_img / latents_std + latents_mean
        # Remove frame dimension: (B, C, 1, H', W') -> (B, C, H', W')
        latents_img = latents_img[:, :, 0]
        image = latents_img  # This is what train_on_df sees as out.images

    else:
        # Original decode path for pixel output
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = pipe.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    pipe.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return QwenImagePipelineOutput(images=image)

