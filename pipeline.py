"""predict(pil_image) -> dict(type, confidence, angle, needs_review). Same logic as classify_and_flag.py."""
import joblib, numpy as np
from embed import embed_image

_art = None

def predict(img):
    global _art
    if _art is None: _art = joblib.load("outputs/classifier.joblib")
    clf = _art["model"]
    P = clf.predict_proba(embed_image(img))          # (4 angles, classes)
    a = int(P.max(1).argmax())                        # most confident angle
    typ, conf = str(clf.classes_[P[a].argmax()]), float(P[a].max())
    return dict(type=typ, confidence=conf, angle=(0, 90, 180, 270)[a],
                needs_review=typ == "Unknown" or conf < _art["threshold"])

if __name__ == "__main__":
    import pandas as pd
    from datasets import load_dataset
    ds = load_dataset("getomni-ai/ocr-benchmark", split="test").select_columns(["id", "image"])
    pr = pd.read_csv("outputs/predictions.csv"); pr = pr[pr.split == "test"].head(10)
    ids = list(ds["id"]); ok = 0
    for _, r in pr.iterrows():
        o = predict(ds[ids.index(r["id"])]["image"])
        same = o["type"] == r.predicted and o["angle"] == r.angle and o["needs_review"] == r.needs_review
        ok += same
        print(f"{r['id']} true={r.true:28s} csv=({r.predicted}, {r.confidence}, {r.angle}) live=({o['type']}, {o['confidence']:.3f}, {o['angle']}) {'OK' if same else 'DIFF'}")
    print(ok, "/ 10 match")
