"""Pipeline + extraction over a split -> results.csv (written after every document; re-runs skip finished ids).
Usage: python run_all.py [--split test] [--limit N] [--out results.csv] [--no-extract]
Extraction: extract_grok.extract with schemas/<predicted class>.json. Env (.env): LLM_API_KEY, LLM_MODEL."""
import argparse, json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from pipeline import predict
from extract_grok import extract, make_client, field_accuracy

EMPTY = 0.30   # filled ratio below this -> human review

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="results.csv"); ap.add_argument("--no-extract", action="store_true"); a = ap.parse_args(argv)
    out = Path(a.out); df = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = set(df["id"]) if len(df) else set()
    sp = pd.read_csv("data/embeddings/split.csv"); sp = sp[(sp.split == a.split) & ~sp.id.isin(done)]
    if a.limit: sp = sp.head(a.limit)
    if not len(sp): print("nothing new to run")
    else:
        ds = load_dataset("getomni-ai/ocr-benchmark", split="test"); row = {i: k for k, i in enumerate(ds["id"])}
        client = None if a.no_extract else make_client()
        rows = df.to_dict("records")
        for n, (_, r) in enumerate(sp.iterrows(), 1):
            d = ds[row[r["id"]]]; p = predict(d["image"])
            o = dict(id=r["id"], split=a.split, true=r["label"], predicted=p["type"], confidence=round(p["confidence"], 3),
                     review_reason="predicted Unknown" if p["type"] == "Unknown" else "low confidence" if p["needs_review"] else "",
                     json_valid=None, filled=None, field_accuracy=None)
            if client and p["type"] != "Unknown":     # Unknown has no schema: straight to human review
                schema = json.loads(Path(f"schemas/{p['type']}.json").read_text())
                try:
                    e = extract(d["image"], schema, client)
                    truth = d["true_json_output"]; truth = json.loads(truth) if isinstance(truth, str) else truth
                    o.update(json_valid=e["valid"], filled=round(e["filled"], 2), field_accuracy=field_accuracy(e["data"], truth))
                except Exception as ex:               # keep going; counts as an empty extraction
                    o.update(json_valid=False, filled=0.0); print("  extraction failed:", type(ex).__name__)
                if o["filled"] is not None and o["filled"] < EMPTY and not o["review_reason"]: o["review_reason"] = "extraction empty"
            o["needs_review"] = bool(o["review_reason"])
            rows.append(o); pd.DataFrame(rows).to_csv(out, index=False)   # save after every doc
            print(f"{n}/{len(sp)}", {k: o[k] for k in ("id", "true", "predicted", "confidence", "review_reason", "json_valid", "field_accuracy")})
        df = pd.DataFrame(rows)
    summary(df)

def summary(df):
    n = len(df); x = df.dropna(subset=["json_valid"])
    print(f"\n=== {n} documents ===\nclassification macro-F1 : {f1_score(df.true, df.predicted, average='macro'):.3f}")
    rv = df.review_reason.fillna(""); print(f"sent to review          : {(rv != '').mean():.1%} ({(rv != '').sum()}/{n})")
    for k in ("low confidence", "predicted Unknown", "extraction empty"): print(f"   - {k:18s}: {(rv == k).sum()} ({(rv == k).mean():.1%})")
    if len(x): print(f"JSON validity rate      : {x.json_valid.astype(bool).mean():.1%} ({len(x)} extracted)\nmean field accuracy     : {x.field_accuracy.mean():.3f}")
    else: print("JSON validity / field accuracy: n/a (no extraction run)")

if __name__ == "__main__": main()
