"""Try extraction on a few documents per class.
Usage: python test_extraction.py data/embeddings --per-class 3 [--split val]
Needs: schemas/ (from build_schemas.py), XAI_API_KEY, GROK_MODEL
"""
import json, argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from extract_grok import extract, field_accuracy, make_client

ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--per-class", type=int, default=3)
ap.add_argument("--split", default="val"); a = ap.parse_args()
sp = pd.read_csv(Path(a.cache) / "split.csv"); ds = load_dataset("getomni-ai/ocr-benchmark", split="test")
client = make_client(); rows = []
for cls, g in sp[(sp.split == a.split) & (sp.label != "Unknown")].groupby("label"):
    schema = json.loads((Path("schemas") / f"{cls}.json").read_text())
    for i in g["id"].head(a.per_class):
        d = ds[int(i)]; r = extract(d["image"], schema, client)
        truth = json.loads(d["true_json_output"]) if isinstance(d["true_json_output"], str) else d["true_json_output"]
        rows.append(dict(id=i, cls=cls, valid=r["valid"], filled=round(r["filled"], 2), field_acc=field_accuracy(r["data"], truth),
                         first_error=(r["errors"] or [""])[0][:120]))
        print(rows[-1])
df = pd.DataFrame(rows); df.to_csv("outputs/extraction_test.csv", index=False)
print("\nPER CLASS"); print(df.groupby("cls").agg(n=("id", "size"), valid=("valid", "mean"), filled=("filled", "mean"), field_acc=("field_acc", "mean")).round(2).to_string())
print("\nOVERALL valid %.0f%% | mean field accuracy %.2f" % (100 * df.valid.mean(), df.field_acc.mean()))
