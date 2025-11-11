from huggingface_hub import snapshot_download
from qwen2vl_flux.model import FluxModel
import os
from dotenv import load_dotenv
import torch
from PIL import Image

# get the HuggingFace access token
load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")

# Download model checkpoints from Hugging Face
snapshot_download("Djrango/Qwen2vl-Flux",
    token=hf_access_token,
    local_dir="../scratch/checkpoints/qwen2vl_flux"
)

structural_prompt = "Keep all other elements, composition, and style unchanged."

def generate_image(input_image, prompt):
    # Text-Guided Blending mode of Qwen2VL-Flux
    complete_prompt = prompt + " " + structural_prompt
    model = FluxModel(device="cuda")
    torch.cuda.empty_cache()
    
    outputs = model.generate(
        input_image_a=input_image,
        prompt=prompt,
        mode="variation",
        guidance_scale=7.5,
        num_inference_steps=28,
        aspect_ratio="784x784"
    )

    # free up CUDA memory
    del model
    torch.cuda.empty_cache()

    return outputs

def save_images(prefix, images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{prefix}_sample{i}.jpg")
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