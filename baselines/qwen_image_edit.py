import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from PIL import Image
import os


model_id = "Qwen/Qwen-Image-Edit"
torch_dtype = torch.bfloat16
device = "cuda"
cache_dir = "../scratch/qwen_image_edit/"

quantization_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
)
transformer = transformer.to("cpu")

quantization_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
)
text_encoder = text_encoder.to("cpu")

pipe = QwenImageEditPipeline.from_pretrained(
    model_id,
    transformer=transformer, text_encoder=text_encoder,
    torch_dtype=torch_dtype, cache_dir=cache_dir,
)

# optionally load LoRA weights to speed up inference
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors",
    cache_dir=cache_dir
)
# pipe.load_lora_weights(
#     "lightx2v/Qwen-Image-Lightning",
#     weight_name="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors",
#     cache_dir=cache_dir,
# )
pipe.enable_model_cpu_offload()

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

        # change steps to 8 or 4 if you used the lighting loras
        torch.cuda.empty_cache()
        with torch.inference_mode():
            image = pipe(input_image, prompt, num_inference_steps=8).images[0]
        image.save(f"{output_dir}/{file_name}.jpg")