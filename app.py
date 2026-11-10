"""streamlit run app.py  -- upload a scan -> type, confidence, auto/review badge, extracted fields."""
import json
from pathlib import Path
import streamlit as st
from PIL import Image
from pipeline import predict
from extract_grok import extract

st.title("Document intake")
f = st.file_uploader("Scanned document", type=["png", "jpg", "jpeg", "tif", "tiff"])
if f:
    img = Image.open(f); p = predict(img)
    left, right = st.columns(2); left.image(img, use_container_width=True)
    right.subheader(p["type"]); right.write(f"confidence {p['confidence']:.2f} · rotation {p['angle']}°")
    (right.error if p["needs_review"] else right.success)("NEEDS REVIEW" if p["needs_review"] else "AUTO")
    if p["type"] == "Unknown": right.info("Out of scope: no extraction, sent to human review.")
    else:
        with st.spinner("Extracting..."):
            e = extract(img, json.loads(Path(f"schemas/{p['type']}.json").read_text()))
        if e["filled"] < 0.30: right.error("NEEDS REVIEW: extraction empty")
        right.write(f"JSON valid ✅ · filled {e['filled']:.0%}" if e["valid"] else f"JSON invalid ❌ {e['errors'][:1]}"); right.json(e["data"])
