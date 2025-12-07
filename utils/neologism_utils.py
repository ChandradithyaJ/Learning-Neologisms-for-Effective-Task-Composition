import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
import torchvision.transforms.functional as TF
from PIL import Image
import io
import ast

def parse_img_field(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unexpected type for image field: {type(value)}")

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

class ClipModel():
    def __init__(self, device='cpu'):
        clip_model_id = "openai/clip-vit-large-patch14"

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)  # we don't train CLIP

        # CLIP normalization constants (standard OpenAI CLIP)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

        self.device = device

    def clip_encode_from_tensor(self, img_tensor, no_grad=False):
        """
        img_tensor: [B, 3, H, W] in [0,1]
        Returns L2-normalized CLIP image embeddings [B, D].
        """
        # Resize to CLIP resolution
        img = F.interpolate(
            img_tensor, size=(224, 224), mode="bicubic", align_corners=False
        )
        img = (img - self.mean) / self.std  # normalize

        if no_grad:
            with torch.no_grad():
                emb = self.clip_model.get_image_features(pixel_values=img)
        else:
            with torch.set_grad_enabled(True):
                emb = self.clip_model.get_image_features(pixel_values=img)

        emb = F.normalize(emb, dim=-1)
        return emb

    def clip_encode_from_pil(self, img):
        """
        Convenience wrapper for target/reject images (no grad needed).
        """
        t = TF.to_tensor(img).unsqueeze(0).to(self.device)  # [1,3,H,W] in [0,1]
        return self.clip_encode_from_tensor(t, no_grad=True)

    def clip_encode_text(self, text, no_grad=False):
        """
        Encode text using the same CLIPModel as the image encoder.
        Returns normalized CLIP text embeddings of shape [B, D].
        """
        # Tokenize with proper CLIP settings
        tokens = self.clip_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        if no_grad:
            with torch.no_grad():
                emb = self.clip_model.get_text_features(**tokens)
        else:
            with torch.set_grad_enabled(True):
                emb = self.clip_model.get_text_features(**tokens)

        # Normalize to match image encoding behavior
        emb = F.normalize(emb, dim=-1)

        return emb