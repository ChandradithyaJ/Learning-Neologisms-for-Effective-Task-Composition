import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import QwenImageEditPipeline
from PIL import Image
import os
import time
from utils.logging_utils import update_csv

model_path = "ovedrive/qwen-image-edit-4bit"

pipeline = QwenImageEditPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    cache_dir="../scratch/qwen_image_edit_4bit"
)
pipeline.set_progress_bar_config(disable=None)
pipeline.enable_model_cpu_offload()

if __name__ == "__main__":
    generator = torch.Generator(device="cpu").manual_seed(42)

    # use test folder
    use_test = False
    if use_test:
        file_name = "test_image_input"

        input_image = Image.open(f"./test/images/{file_name}.jpg").convert('RGB')
        prompt = open(f"./test/prompts/{file_name}.txt", 'r').read()
        output_dir = f"./test/outputs"
        os.makedirs(output_dir, exist_ok=True)

        torch.cuda.empty_cache()
        with torch.inference_mode():
            image = pipeline(
                image=input_image,
                prompt=prompt,
                generator=generator,
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=10
            ).images[0]
        image.save(f"{output_dir}/{file_name}.jpg")

    else:
        path_to_images = '../scratch/DL_data/images'
        path_to_prompt = '../scratch/DL_data/prompts/composite'

        input_path = f'{path_to_images}/original'
        output_dir = f'{path_to_images}/qwen_image_edit_output'
        os.makedirs(output_dir, exist_ok=True)

        csv_path = f'{output_dir}/times.csv'

        for fname in os.listdir(input_path):
            start_time = time.time()

            file_name = fname[:-4] # removes ".png"

            input_image = Image.open(f"{input_path}/{file_name}.png").convert('RGB')
            prompt = open(f"{path_to_prompt}/{file_name}.txt", 'r').read()

            torch.cuda.empty_cache()

            with torch.inference_mode():
                image = pipeline(
                    image=input_image,
                    prompt=prompt,
                    generator=generator,
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=10
                ).images[0]
                image.save(f"{output_dir}/{file_name}.png")

            torch.cuda.empty_cache()

            time_taken = time.time() - start_time
            update_csv(csv_path, file_name, time_taken)
            print(f"Edited image saved to {output_dir}/{file_name}.png | Time: {time_taken}s")

            break