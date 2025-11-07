import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from dotenv import load_dotenv
import os
# os.environ['HF_HOME'] = './checkpoints/flux1kontext'

# get the HuggingFace access token
load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", 
    torch_dtype=torch.bfloat16,
    token=hf_access_token,
    cache_dir='../scratch/checkpoints/flux1kontext',
    resume_download=True
)
pipe.to("cuda")

def generate_image(input_image, prompt):
    output_image = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=2.5
    ).images[0]
    return output_image

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
        input_image = open(f'../test/{file_name}.jpg', 'rb')
        with open('../test/prompt.txt', 'rb') as f:
            prompt = f.read()
        output_dir = "../test/output"

        output_images = generate_image(input_image, prompt)
        save_images(file_name, output_images, output_dir)