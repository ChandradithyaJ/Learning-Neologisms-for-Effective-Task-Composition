from huggingface_hub import snapshot_download
from qwen2vl_flux.model import FluxModel
import os
from dotenv import load_dotenv

# get the HuggingFace access token
load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")

# Download model checkpoints from Hugging Face
snapshot_download("Djrango/Qwen2vl-Flux",
    token=hf_access_token,
    local_dir="./checkpoints/qwen2vl_flux"
)

# instantiate the model
model = FluxModel(device="cuda")

def generate_image(input_image, prompt):
    # Text-Guided Blending mode of Qwen2VL-Flux
    outputs = model.generate(
        input_image_a=input_image,
        prompt=prompt,
        mode="variation",
        guidance_scale=7.5
    )
    return outputs

if __name__ == "__main__":
    use_test = True # use the test folder examples

    if use_test:
        input_image = open('../test/test_image_input.jpg', 'rb')
        with open('../test/prompt.txt', 'rb') as f:
            prompt = f.read()

        output_image = generate_image(input_image, prompt)

        print(output_image)