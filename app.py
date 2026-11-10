"""streamlit run app.py  -- upload a scan -> type, confidence, auto/review badge, extracted fields."""
import json, os
import pandas as pd, streamlit as st
from PIL import Image
from pipeline import predict
from extract import extract

@st.cache_data
def type_schema(label):
    # shortcut: schemas are per-document in this dataset; use the first train doc's schema of the type. Exact only for single-schema types.
    from datasets import load_dataset
    sp = pd.read_csv("data/embeddings/split.csv"); i = sp[(sp.split == "train") & (sp.label == label)].index[0]
    return load_dataset("getomni-ai/ocr-benchmark", split="test")[int(i)]["json_schema"]

st.title("Document intake")
f = st.file_uploader("Scanned document", type=["png", "jpg", "jpeg", "tif", "tiff"])
if f:
    img = Image.open(f); p = predict(img)
    left, right = st.columns(2); left.image(img, use_container_width=True)
    right.subheader(p["type"]); right.write(f"confidence {p['confidence']:.2f} · rotation {p['angle']}°")
    (right.error if p["needs_review"] else right.success)("NEEDS REVIEW" if p["needs_review"] else "AUTO")
    if p["type"] == "Unknown": right.info("Out of scope: no extraction.")
    elif not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_MODEL")): right.warning("Set ANTHROPIC_API_KEY and CLAUDE_MODEL to enable extraction.")
    else:
        with st.spinner("Extracting..."): e = extract(img, type_schema(p["type"]))
        right.write("JSON valid ✅" if e["valid"] else f"JSON invalid ❌ {e['error']}"); right.json(e["result"])
