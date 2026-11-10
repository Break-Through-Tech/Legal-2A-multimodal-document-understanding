"""Try extraction on a few documents per class.
Usage: python test_extraction.py data/embeddings --per-class 3 [--split val]
Needs: schemas/ (from build_schemas.py), LLM_API_KEY and LLM_MODEL in .env
"""
import json, argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from extract_grok import extract, field_accuracy, value_recall, make_client

ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--per-class", type=int, default=3)
ap.add_argument("--split", default="val"); a = ap.parse_args()
sp = pd.read_csv(Path(a.cache) / "split.csv"); ds = load_dataset("getomni-ai/ocr-benchmark", split="test")
client = make_client(); rows = []
for cls, g in sp[(sp.split == a.split) & (sp.label != "Unknown")].groupby("label"):
    schema = json.loads((Path("schemas") / f"{cls}.json").read_text())
    for i in g["id"].head(a.per_class):
        d = ds[int(i)]
        try:
            r = extract(d["image"], schema, client)
            truth = json.loads(d["true_json_output"]) if isinstance(d["true_json_output"], str) else d["true_json_output"]
            rows.append(dict(id=i, cls=cls, valid=r["valid"], filled=round(r["filled"], 2), field_acc=field_accuracy(r["data"], truth), value_recall=value_recall(r["data"], truth),
                             first_error=(r["errors"] or [""])[0][:120]))
        except Exception as e:      # one bad call must not kill the whole run
            rows.append(dict(id=i, cls=cls, valid=False, filled=0.0, field_acc=None, value_recall=None, first_error=("CALL FAILED: " + str(e))[:120]))
        print(rows[-1]); pd.DataFrame(rows).to_csv("outputs/extraction_test.csv", index=False)
df = pd.DataFrame(rows); df.to_csv("outputs/extraction_test.csv", index=False)
print("\nPER CLASS"); print(df.groupby("cls").agg(n=("id", "size"), valid=("valid", "mean"), filled=("filled", "mean"), field_acc=("field_acc", "mean"), value_recall=("value_recall", "mean")).round(2).to_string())
print("\nOVERALL valid %.0f%% | mean field accuracy %.2f | mean value recall %.2f" % (100 * df.valid.mean(), df.field_acc.mean(), df.value_recall.mean()))