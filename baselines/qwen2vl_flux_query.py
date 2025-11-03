from huggingface_hub import snapshot_download
from qwen2vl_flux.model import FluxModel
import os
from dotenv import load_dotenv

# get the HuggingFace access token
load_dotenv()
hf_access_token = os.getenv("HF_ACCESS_TOKEN")

# Download model checkpoints from Hugging Face
snapshot_download("Djrango/Qwen2vl-Flux", token=hf_access_token)