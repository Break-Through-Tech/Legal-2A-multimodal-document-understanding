"""embed_image(pil) -> (4, 768) L2-normalised SigLIP embeddings for rotations 0/90/180/270."""
import numpy as np, torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

SIGLIP_ID = "google/siglip-base-patch16-224"
_model = _proc = None

def _load():
    global _model, _proc
    if _model is None:
        _model = AutoModel.from_pretrained(SIGLIP_ID).eval()   # fp32 on CPU (cache was fp16 on GPU)
        _proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    return _model, _proc

def _prep(img):  # same as notebooks/01_siglip_embeddings.ipynb: RGB + white square pad
    img = img.convert("RGB"); s = max(img.size)
    return ImageOps.pad(img, (s, s), color="white")

def embed_image(img):
    model, proc = _load()
    imgs = [_prep(img.rotate(a, expand=True)) for a in (0, 90, 180, 270)]   # PIL rotate = counter-clockwise
    with torch.inference_mode():
        f = model.get_image_features(pixel_values=proc(images=imgs, return_tensors="pt")["pixel_values"])
        if not torch.is_tensor(f): f = f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).numpy()

if __name__ == "__main__":  # self-check against cached vectors
    import pandas as pd
    from datasets import load_dataset
    ds = load_dataset("getomni-ai/ocr-benchmark", split="test").select_columns(["id", "image"])
    R = np.load("data/embeddings/siglip_rot4.npy"); R = R / np.linalg.norm(R, axis=-1, keepdims=True)
    for i in range(0, 1000, 200):
        e = embed_image(ds[i]["image"]); cos = (e * R[:, i]).sum(1)
        print(i, "cosine per angle:", cos.round(4)); assert cos[0] > 0.999, "angle 0 mismatch"
