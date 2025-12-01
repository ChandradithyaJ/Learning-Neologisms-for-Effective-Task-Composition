import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import time
from utils.logging import update_csv

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, 
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

if __name__ == "__main__":
    path_to_images = '../scratch/DL_data/images'
    path_to_prompt = '../scratch/DL_data/prompts/CoT'

    input_path = f'{path_to_images}/original'
    output_dir = f'{path_to_images}/instructpix2pix_cot_output'
    os.makedirs(output_dir, exist_ok=True)

    csv_path = f'{output_dir}/times.csv'

    for fname in os.listdir(input_path):
        start_time = time.time()

        file_name = fname[:-4] # removes ".png"

        input_image = Image.open(f"{input_path}/{file_name}.png").convert('RGB')
        prompt = open(f"{path_to_prompt}/{file_name}.txt", 'r').read()

        torch.cuda.empty_cache()

        subprompts = prompt.split("||")
        with torch.inference_mode():
            image = input_image
            for prompt in subprompts:
                image = pipe(prompt, 
                    image=image, 
                    num_inference_steps=20,
                    image_guidance_scale=1
                ).images[0]
            image.save(f"{output_dir}/{file_name}.png")

        torch.cuda.empty_cache()

        time_taken = time.time() - start_time
        update_csv(csv_path, file_name, time_taken)
        print(f"Edited image saved to {output_dir}/{file_name}.png | Time: {time_taken}s")

        break