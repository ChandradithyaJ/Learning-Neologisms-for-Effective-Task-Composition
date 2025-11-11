import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import os
from dotenv import load_dotenv

# get the HuggingFace access token
load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")

pipe = DiffusionPipeline.from_pretrained("sanaka87/ICEdit-MoE-LoRA", 
    dtype=torch.bfloat16,
    device_map="cuda",
    safety_checker=None,
    token=hf_access_token,
    local_dir="../scratch/checkpoints/icedit"
)

def generate_image(input_image, prompt):
    output_image = pipe(image=input_image, prompt=prompt).images[0]
    return output_image

def save_image(prefix, image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}.jpg")
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

        output_image = generate_image(input_image, prompt)
        save_image(file_name, output_image, output_dir)