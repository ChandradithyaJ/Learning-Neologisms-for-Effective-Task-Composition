import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from dotenv import load_dotenv
import os
from PIL import Image

# get the HuggingFace access token
load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")

# pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", 
#     torch_dtype=torch.bfloat16,
#     token=hf_access_token,
#     cache_dir='../scratch/checkpoints/flux1kontext',
# )
pipe = FluxKontextPipeline.from_pretrained("kpsss34/FLUX.1-Kontext-dev-int4", 
    torch_dtype=torch.bfloat16,
    token=hf_access_token,
    cache_dir='../scratch/checkpoints/flux1kontext-int4'
)
pipe.to("cuda")

def generate_image(input_image, prompt):
    output_image = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=2.5
    ).images
    return output_images

def save_images(prefix, images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{prefix}_{i+1}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")

if __name__ == "__main__":
    use_test = True # use the test folder examples

    if use_test:
        file_name = "test_image_input"
        input_image = Image.open(f'./test/images/{file_name}.jpg')
        with open(f'./test/prompts/{file_name}.txt', 'r') as f:
            prompt = f.read()
        output_dir = "./test/output"

        output_images = generate_image(input_image, prompt)
        save_images(file_name, output_images, output_dir)