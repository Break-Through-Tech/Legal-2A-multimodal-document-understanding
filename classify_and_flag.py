"""Step 1 of the pipeline: classify each document and decide 'auto' vs 'human review'.
Usage:  python classify_and_flag.py "/path/to/legal_idp/embeddings"
Needs in that folder: split.csv, ids.npy, siglip_siglip-base-patch16-224.npy
Optional (used if present): siglip_rot4.npy  -> test-time rotation
"""
import sys, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

CACHE = Path(sys.argv[1]); SEED = 42
C, THRESH = 100, 0.80            # first working values, tune later
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

sp = pd.read_csv(CACHE / "split.csv"); y = sp["label"].to_numpy()
assert (sp["id"].to_numpy() == np.load(CACHE / "ids.npy")).all(), "split.csv and ids.npy are out of order"
unit = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)
if (CACHE / "siglip_rot4.npy").exists():
    R = unit(np.load(CACHE / "siglip_rot4.npy")); print("using 4 rotations")
else:
    R = unit(np.load(CACHE / "siglip_siglip-base-patch16-224.npy"))[None]; print("WARNING: no siglip_rot4.npy, rotation step skipped")
tr, va, te = [(sp["split"] == s).to_numpy() for s in ("train", "val", "test")]

clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED).fit(R[0][tr], y[tr])

def predict(use_rotation):
    P = np.stack([clf.predict_proba(R[a]) for a in (range(len(R)) if use_rotation else [0])])   # (angles, docs, classes)
    best = P.max(2).argmax(0)                                   # angle with the most confident answer
    p = P[best, np.arange(P.shape[1])]
    return clf.classes_[p.argmax(1)], p.max(1), best

def report(name, pred, conf, idx):
    review = (pred[idx] == "Unknown") | (conf[idx] < THRESH)
    oos = y[idx] == "Unknown"
    print(f"{name:34s} macroF1 {f1_score(y[idx], pred[idx], average='macro'):.3f} | acc {accuracy_score(y[idx], pred[idx]):.3f} | "
          f"out-of-scope caught {review[oos].mean():.0%} ({review[oos].sum()}/{oos.sum()}) | "
          f"in-scope sent to human {review[~oos].mean():.1%} ({review[~oos].sum()}/{(~oos).sum()})")

for use_rot in ([False, True] if len(R) == 4 else [False]):
    pred, conf, best = predict(use_rot)
    tag = "with test-time rotation" if use_rot else "no rotation"
    report(f"VAL  {tag}", pred, conf, va); report(f"TEST {tag}", pred, conf, te)

# final choice for the demo: rotation on if available
USE_ROT = len(R) == 4
pred, conf, best = predict(USE_ROT)
out = sp[["id", "split", "label"]].rename(columns={"label": "true"}).assign(
    predicted=pred, confidence=conf.round(3), angle=[(0, 90, 180, 270)[b] for b in best])
out["needs_review"] = (out["predicted"] == "Unknown") | (out["confidence"] < THRESH)
out.to_csv(OUT / "predictions.csv", index=False)
joblib.dump(dict(model=clf, threshold=THRESH, use_rotation=USE_ROT), OUT / "classifier.joblib")
print("\nsaved outputs/predictions.csv and outputs/classifier.joblib")
print("test docs sent to human:", int(out[out.split == "test"].needs_review.sum()), "of", int((out.split == "test").sum()))