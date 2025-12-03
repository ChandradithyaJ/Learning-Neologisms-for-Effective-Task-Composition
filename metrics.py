from ignite.metrics import SSIM, PSNR
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def compute_ssim(generated_images, ground_truth_images):
    """
    Requires ground truth images
    Pass image paths
    https://docs.pytorch.org/ignite/generated/ignite.metrics.SSIM.html
    """
    to_tensor = transforms.ToTensor()
    ssim = SSIM(data_range=1.0)

    for gen_img, gt_img in zip(generated_images, ground_truth_images):
        gen_img = Image.open(gen_img)
        gt_img = Image.open(gt_img)

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
        gen_img = Image.open(gen_img)
        gt_img = Image.open(gt_img)

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