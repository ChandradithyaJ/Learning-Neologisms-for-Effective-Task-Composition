from ignite.metrics import SSIM, PSNR
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import open_clip
import os

FIXED_RESOLUTION = (256, 256)
resize_transform = transforms.Resize(FIXED_RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC)

def compute_ssim(generated_images, ground_truth_images):
    """
    Requires ground truth images
    Pass image paths
    https://docs.pytorch.org/ignite/generated/ignite.metrics.SSIM.html
    """
    to_tensor = transforms.ToTensor()
    ssim = SSIM(data_range=1.0)

    for gen_img, gt_img in zip(generated_images, ground_truth_images):
        gen_img = resize_transform(Image.open(gen_img).convert("RGB"))
        gt_img  = resize_transform(Image.open(gt_img).convert("RGB"))

        gen_img = to_tensor(gen_img).unsqueeze(0)
        gt_img = to_tensor(gt_img).unsqueeze(0)

        ssim.update((gen_img, gt_img))

    avg_ssim = ssim.compute()
    return avg_ssim

def compute_psnr(generated_images, ground_truth_images):
    """
    Requires ground truth images
    Pass image paths
    https://docs.pytorch.org/ignite/generated/ignite.metrics.PSNR.html
    """
    to_tensor = transforms.ToTensor()
    psnr = PSNR(data_range=1.0)

    for gen_img, gt_img in zip(generated_images, ground_truth_images):
        gen_img = resize_transform(Image.open(gen_img).convert("RGB"))
        gt_img  = resize_transform(Image.open(gt_img).convert("RGB"))

        gen_img = to_tensor(gen_img).unsqueeze(0)
        gt_img = to_tensor(gt_img).unsqueeze(0)

        psnr.update((gen_img, gt_img))

    avg_psnr = psnr.compute()
    return avg_psnr

def DINO_similarity(generated_images, ground_truth_images):
    model = torch.hub.load(
        'facebookresearch/dinov2', 
        'dinov2_vitb14'
    )
    model.eval.cuda()

    # required preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    gen_imgs = []
    gt_imgs = []
    
    for gen_path, gt_path in zip(generated_images, ground_truth_images):
        gen_img = Image.open(gen_path)
        gt_img = Image.open(gt_path)
        
        gen_imgs.append(transform(gen_img))
        gt_imgs.append(transform(gt_img))
    
    gen_batch = torch.stack(gen_imgs).cuda()
    gt_batch = torch.stack(gt_imgs).cuda()  
    
    with torch.no_grad():
        gen_features = F.normalize(model(gen_batch), dim=-1)
        gt_features = F.normalize(model(gt_batch), dim=-1) 
    
    similarities = (gen_features * gt_features).sum(dim=-1)
    
    avg_similarity = similarities.mean().item()
    return avg_similarity


def CLIP_similarity(generated_images, ground_truth_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Correct usage: separate model name and pretrained weights
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    gen_tensors = []
    gt_tensors = []

    for gen_path, gt_path in zip(generated_images, ground_truth_images):
        gen_img = preprocess(Image.open(gen_path).convert("RGB"))
        gt_img  = preprocess(Image.open(gt_path).convert("RGB"))

        gen_tensors.append(gen_img)
        gt_tensors.append(gt_img)

    gen_batch = torch.stack(gen_tensors).to(device)
    gt_batch  = torch.stack(gt_tensors).to(device)

    with torch.no_grad():
        gen_features = model.encode_image(gen_batch)
        gt_features  = model.encode_image(gt_batch)

        gen_features = F.normalize(gen_features, dim=-1)
        gt_features  = F.normalize(gt_features, dim=-1)

    similarities = (gen_features * gt_features).sum(dim=-1)  # cosine similarity
    avg_similarity = similarities.mean().item()
    return avg_similarity

def CLIP_direction_similarity(generated_images, original_images, edit_prompts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Correct usage: separate model name and pretrained weights
    model_name = "ViT-B-32"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    gen_tensors = []
    orig_tensors = []
    prompts = []

    for gen_path, orig_path, prompt_path in zip(generated_images, original_images, edit_prompts):
        gen_img = preprocess(Image.open(gen_path).convert("RGB"))
        orig_img  = preprocess(Image.open(orig_path).convert("RGB"))
        prompt = open(prompt_path, 'r').read()

        gen_tensors.append(gen_img)
        orig_tensors.append(orig_img)
        prompts.append(prompt)

    gen_batch = torch.stack(gen_tensors).to(device)
    orig_batch  = torch.stack(orig_tensors).to(device)

    with torch.no_grad():
        gen_features = model.encode_image(gen_batch)
        orig_features  = model.encode_image(orig_batch)

        gen_features = F.normalize(gen_features, dim=-1)
        orig_features  = F.normalize(orig_features, dim=-1)

    # compute delta image direction
    delta_img = gen_features - orig_features
    delta_img = F.normalize(delta_img, dim=-1)

    # encode prompts
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    # directional similarity
    similarities = (delta_img * text_features).sum(dim=-1)
    avg_similarity = similarities.mean().item()
    return avg_similarity

if __name__ == "__main__":
    generated_images_folder = "../scratch/DL_data/images/instruct_pix2pix_outputs_neologism_and_1stepsPerImage_80trainImages_100epochs_8denoisingSteps_ckpt70"
    ground_truth_images_folder = "../scratch/DL_data/images/final"
    original_images_folder = "../scratch/DL_data/images/original"
    edit_prompts_folder = "../scratch/DL_data/prompts/composite"

    generated_images = sorted(os.listdir(generated_images_folder))
    ground_truth_images = sorted(os.listdir(ground_truth_images_folder))
    original_images = sorted(os.listdir(original_images_folder))
    edit_prompts = sorted(os.listdir(edit_prompts_folder))

    generated_images = [os.path.join(generated_images_folder, f) for f in generated_images]
    ground_truth_images = [os.path.join(ground_truth_images_folder, f) for f in ground_truth_images]
    original_images = [os.path.join(ground_truth_images_folder, f) for f in original_images]
    edit_prompts = [os.path.join(edit_prompts_folder, f) for f in edit_prompts]

    ssim = compute_ssim(generated_images, ground_truth_images)
    psnr = compute_psnr(generated_images, ground_truth_images)
    dino = DINO_similarity(generated_images, ground_truth_images)
    clip_sim = CLIP_similarity(generated_images, ground_truth_images)
    clip_dir_sim = CLIP_direction_similarity(generated_images, ground_truth_images, edit_prompts)

    print("=" * 20)
    print("Metrics:", generated_images_folder)
    print("SSIM", ssim)
    print("PSNR", psnr)
    print("DINO", dino)
    print("CLIP Similarity", clip_sim)
    print("CLIP Direction Similarity", clip_dir_sim)
    print("=" * 20)