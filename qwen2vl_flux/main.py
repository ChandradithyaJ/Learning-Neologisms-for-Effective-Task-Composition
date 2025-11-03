import argparse
import torch
from PIL import Image
import numpy as np
from model import FluxModel

def parse_args():
    parser = argparse.ArgumentParser(description='Flux Image Generation Tool')
    
    # Required arguments
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['variation', 'img2img', 'inpaint', 'controlnet', 'controlnet-inpaint'],
                      help='Generation mode')
    parser.add_argument('--input_image', type=str, required=True,
                      help='Path to the input image')
    
    # Optional arguments
    parser.add_argument('--prompt', type=str, default="",
                      help='Text prompt to guide the generation')
    parser.add_argument('--reference_image', type=str, default=None,
                      help='Path to the reference image (for img2img/controlnet modes)')
    parser.add_argument('--mask_image', type=str, default=None,
                      help='Path to the mask image (for inpainting modes)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save generated images')
    parser.add_argument('--image_count', type=int, default=1,
                      help='Number of images to generate')
    parser.add_argument('--aspect_ratio', type=str, default='1:1',
                      choices=['1:1', '16:9', '9:16', '2.4:1', '3:4', '4:3'],
                      help='Output image aspect ratio')
    parser.add_argument('--steps', type=int, default=28,
                      help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                      help='Guidance scale for generation')
    parser.add_argument('--denoise_strength', type=float, default=0.8,
                      help='Denoising strength for img2img/inpaint')
    
    # Attention related arguments
    parser.add_argument('--center_x', type=float, default=None,
                      help='X coordinate of attention center (0-1)')
    parser.add_argument('--center_y', type=float, default=None,
                      help='Y coordinate of attention center (0-1)')
    parser.add_argument('--radius', type=float, default=None,
                      help='Radius of attention circle (0-1)')
    
    # ControlNet related arguments
    parser.add_argument('--line_mode', action='store_true',
                      help='Enable line detection mode for ControlNet')
    parser.add_argument('--depth_mode', action='store_true',
                      help='Enable depth mode for ControlNet')
    parser.add_argument('--line_strength', type=float, default=0.4,
                      help='Strength of line guidance')
    parser.add_argument('--depth_strength', type=float, default=0.2,
                      help='Strength of depth guidance')
    
    # Device selection
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to run the model on')
    parser.add_argument('--turbo', action='store_true',
                      help='Enable turbo mode for faster inference')
    
    return parser.parse_args()

def load_image(image_path):
    """Load and return a PIL Image."""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")

def save_images(images, output_dir, prefix="generated"):
    """Save generated images with sequential numbering."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{prefix}_{i+1}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")

def get_required_features(args):
    """Determine which model features are required based on the arguments."""
    features = []
    
    if args.mode in ['controlnet', 'controlnet-inpaint']:
        features.append('controlnet')
        if args.depth_mode:
            features.append('depth')
        if args.line_mode:
            features.append('line')
            
    if args.mode in ['inpaint', 'controlnet-inpaint']:
        features.append('sam')  # If you're using SAM for mask generation
        
    return features


def main():
    args = parse_args()
    
    # Check CUDA availability if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Determine required features based on mode and arguments
    required_features = get_required_features(args)
    
    # Initialize model with only required features
    print(f"Initializing model on {args.device} with features: {required_features}")
    model = FluxModel(
        is_turbo=args.turbo,
        device=args.device,
        required_features=required_features
    )
    
    # Load input images
    input_image = load_image(args.input_image)
    reference_image = load_image(args.reference_image) if args.reference_image else None
    mask_image = load_image(args.mask_image) if args.mask_image else None
    
    # Validate inputs based on mode
    if args.mode in ['inpaint', 'controlnet-inpaint'] and mask_image is None:
        raise ValueError(f"{args.mode} mode requires a mask image")
    
    # Generate images
    print(f"Generating {args.image_count} images in {args.mode} mode...")
    generated_images = model.generate(
        input_image_a=input_image,
        input_image_b=reference_image,
        prompt=args.prompt,
        mask_image=mask_image,
        mode=args.mode,
        imageCount=args.image_count,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        denoise_strength=args.denoise_strength,
        center_x=args.center_x,
        center_y=args.center_y,
        radius=args.radius,
        line_mode=args.line_mode,
        depth_mode=args.depth_mode,
        line_strength=args.line_strength,
        depth_strength=args.depth_strength
    )
    
    # Save generated images
    save_images(generated_images, args.output_dir)
    print("Generation completed successfully!")

if __name__ == "__main__":
    main()