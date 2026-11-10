"""Run pipeline + extraction on the test split -> results.csv + summary.
Usage: python run_all.py [--per-type N] [--no-extract]   (--per-type N: only N test docs per in-scope type, for a quick check)
Extraction uses each document's OWN ground-truth json_schema (the dataset has per-document schemas)."""
import argparse, json, os
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from pipeline import predict
from extract import extract, field_accuracy

ap = argparse.ArgumentParser(); ap.add_argument("--per-type", type=int); ap.add_argument("--no-extract", action="store_true")
a = ap.parse_args()
do_extract = not a.no_extract and bool(os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_MODEL"))
if not a.no_extract and not do_extract: print("!! ANTHROPIC_API_KEY / CLAUDE_MODEL not set -> extraction skipped")

ds = load_dataset("getomni-ai/ocr-benchmark", split="test")
row = {i: k for k, i in enumerate(ds["id"])}
sp = pd.read_csv("data/embeddings/split.csv"); sp = sp[sp.split == "test"]
if a.per_type: sp = sp[sp.label != "Unknown"].groupby("label").head(a.per_type)

rows = []
for n, (_, r) in enumerate(sp.iterrows(), 1):
    d = ds[row[r["id"]]]; p = predict(d["image"])
    o = dict(id=r["id"], true=r["label"], predicted=p["type"], confidence=round(p["confidence"], 3),
             needs_review=p["needs_review"], json_valid=None, field_accuracy=None)
    if do_extract and p["type"] != "Unknown":
        try:
            e = extract(d["image"], d["json_schema"])
            o["json_valid"] = e["valid"]; o["field_accuracy"] = field_accuracy(e["result"], json.loads(d["true_json_output"]))
        except Exception as ex:   # API/network error: record as invalid, keep going
            o["json_valid"], o["field_accuracy"] = False, 0.0; print("extract failed", r["id"], ex)
    rows.append(o); print(n, len(sp), o)

df = pd.DataFrame(rows); out = "outputs/extract_check.csv" if a.per_type else "results.csv"; df.to_csv(out, index=False)
x = df.dropna(subset=["json_valid"])
print(f"\nSaved {out}  ({len(df)} docs)")
print(f"classification macro-F1 : {f1_score(df.true, df.predicted, average='macro'):.3f}")
print(f"sent to review          : {df.needs_review.mean():.1%} ({df.needs_review.sum()}/{len(df)})")
if len(x): print(f"JSON validity rate      : {x.json_valid.astype(bool).mean():.1%} ({len(x)} extracted)\nmean field accuracy     : {x.field_accuracy.mean():.3f}")
else: print("JSON validity / field accuracy: n/a (no extraction run)")
