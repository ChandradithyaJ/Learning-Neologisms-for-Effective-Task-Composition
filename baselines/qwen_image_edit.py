import torch
from diffusers import QwenImageEditPipeline
from PIL import Image
import os

model_path = "ovedrive/qwen-image-edit-4bit"
pipeline = QwenImageEditPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    cache_dir="../scratch/qwen_image_edit"
)
pipeline.set_progress_bar_config(disable=None)
pipeline.enable_model_cpu_offload()

if __name__ == "__main__":
    generator = torch.Generator(device="cpu").manual_seed(42)

    # use test folder
    use_test = True
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
                num_inference_steps=20
            ).images[0]
        image.save(f"{output_dir}/{file_name}.jpg")