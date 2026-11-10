"""Build ONE schema per in-scope class from the dataset's own per-document schemas.
For each class, the schema contains every field that appears in that class's TRAIN documents (union),
so the model may only return fields that class's files actually have.

Usage: python build_schemas.py data/embeddings [--min-freq 0.0]
Writes: schemas/<CLASS>.json and schemas/_report.csv  (class, field path, share of docs having it)
"""
import sys, json, argparse, collections
from pathlib import Path
import numpy as np, pandas as pd

def as_dict(s):
    return json.loads(s) if isinstance(s, str) else s

def types_of(node):
    t = node.get("type")
    return set([t] if isinstance(t, str) else (t or []))

def merge(a, b):
    """Union of two schema nodes (recursive)."""
    if a is None: return json.loads(json.dumps(b))
    if b is None: return json.loads(json.dumps(a))
    out = {}
    ts = types_of(a) | types_of(b)
    if ts: out["type"] = sorted(ts)
    desc = a.get("description") or b.get("description")
    if desc: out["description"] = desc
    if a.get("enum") and b.get("enum"): out["enum"] = sorted(set(map(str, a["enum"])) | set(map(str, b["enum"])))
    if "object" in ts:
        props = {}
        for k in set(a.get("properties", {})) | set(b.get("properties", {})):
            props[k] = merge(a.get("properties", {}).get(k), b.get("properties", {}).get(k))
        out["properties"] = props
    if "array" in ts:
        ia, ib = a.get("items"), b.get("items")
        if ia or ib: out["items"] = merge(ia, ib)
    return out

def paths(node, prefix=""):
    """All property paths in one schema, e.g. 'items[].amount'."""
    res = []
    for k, v in node.get("properties", {}).items():
        p = f"{prefix}.{k}" if prefix else k
        res.append(p); res += paths(v, p)
    if "items" in node and isinstance(node["items"], dict):
        res += paths(node["items"], prefix + "[]")
    return res

def finalize(node, prune=lambda p: False, prefix=""):
    """Make every field optional + nullable, forbid unknown fields (so output stays inside the class schema)."""
    out = dict(node)
    ts = set(types_of(node)); ts.add("null"); out["type"] = sorted(ts)
    if "object" in ts and "properties" in node:
        out["properties"] = {k: finalize(v, prune, f"{prefix}.{k}" if prefix else k)
                             for k, v in node["properties"].items() if not prune(f"{prefix}.{k}" if prefix else k)}
        out["required"] = []; out["additionalProperties"] = False
    if "array" in ts and isinstance(node.get("items"), dict):
        out["items"] = finalize(node["items"], prune, prefix + "[]")
    return out

def build(df_schemas, min_freq=0.0):
    """df_schemas: iterable of schema dicts for ONE class -> (schema, {path: share})."""
    docs = [as_dict(s) for s in df_schemas]
    merged, counts = None, collections.Counter()
    for s in docs:
        merged = merge(merged, s)
        counts.update(set(paths(s)))
    freq = {p: c / len(docs) for p, c in counts.items()}
    schema = finalize(merged, prune=lambda p: freq.get(p, 0) < min_freq)
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    return schema, freq

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--min-freq", type=float, default=0.0)
    ap.add_argument("--splits", default="train", help="which split rows to learn schemas from (default train only: no test leakage)")
    a = ap.parse_args()
    from datasets import load_dataset
    sp = pd.read_csv(Path(a.cache) / "split.csv")
    ds = load_dataset("getomni-ai/ocr-benchmark", split="test").select_columns(["id", "json_schema"])
    Path("schemas").mkdir(exist_ok=True); rows = []
    use = sp[sp.split.isin(a.splits.split(","))]
    for cls, g in use.groupby("label"):
        if cls == "Unknown": continue
        schema, freq = build([ds[int(i)]["json_schema"] for i in g["id"]], a.min_freq)
        (Path("schemas") / f"{cls}.json").write_text(json.dumps(schema, indent=2))
        rows += [dict(cls=cls, path=p, share=round(s, 3), n_docs=len(g)) for p, s in sorted(freq.items())]
        n_top = sum(1 for p in freq if "." not in p and "[]" not in p)
        print(f"{cls:28s} docs {len(g):3d} | fields in schema {len([p for p in freq if freq[p] >= a.min_freq]):4d} | top-level {n_top:3d} | in >=50% of docs: {sum(s >= .5 for s in freq.values())}")
    pd.DataFrame(rows).to_csv("schemas/_report.csv", index=False)
    print("wrote schemas/*.json and schemas/_report.csv")
