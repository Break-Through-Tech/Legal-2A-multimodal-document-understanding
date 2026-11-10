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

VALID = {"string", "number", "integer", "boolean", "object", "array", "null"}

def types_of(node):
    t = node.get("type")
    ts = set([t] if isinstance(t, str) else (t or []))
    out = set()
    for x in ts:                      # the dataset has odd types such as "enum": treat as string, drop anything else unknown
        if x in VALID: out.add(x)
        elif x == "enum": out.add("string")
    return out

def merge(a, b):
    """Union of two schema nodes (recursive)."""
    if a is None: return json.loads(json.dumps(b))
    if b is None: return json.loads(json.dumps(a))
    out = {}
    ts = types_of(a) | types_of(b)
    if ts: out["type"] = sorted(ts)
    desc = a.get("description") or b.get("description")
    if desc: out["description"] = desc
    if a.get("enum") and b.get("enum"):
        seen = {}
        for v in list(a["enum"]) + list(b["enum"]): seen[json.dumps(v)] = v
        out["enum"] = list(seen.values())
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
    if "enum" in out and None not in out["enum"]: out["enum"] = list(out["enum"]) + [None]   # null must stay allowed
    if "object" in ts and "properties" in node:
        out["properties"] = {k: finalize(v, prune, f"{prefix}.{k}" if prefix else k)
                             for k, v in node["properties"].items() if not prune(f"{prefix}.{k}" if prefix else k)}
        out["required"] = []; out["additionalProperties"] = False
    if "array" in ts and isinstance(node.get("items"), dict):
        out["items"] = finalize(node["items"], prune, prefix + "[]")
    return out

def n_kept(schema):
    return len(paths(schema))

def build(df_schemas, min_freq=0.0, max_fields=None):
    """df_schemas: iterable of schema dicts for ONE class -> (schema, {path: share})."""
    docs = [as_dict(s) for s in df_schemas]
    merged, counts = None, collections.Counter()
    for s in docs:
        merged = merge(merged, s)
        counts.update(set(paths(s)))
    freq = {p: c / len(docs) for p, c in counts.items()}
    if max_fields and len(freq) > max_fields:          # keep only the most common fields (a child is never more common than its parent)
        vals = sorted(set(freq.values()))                # smallest cutoff that keeps at most max_fields fields (ties kept together)
        cut = next((t for t in vals if sum(f >= t for f in freq.values()) <= max_fields), vals[-1])
        min_freq = max(min_freq, cut)
    schema = finalize(merged, prune=lambda p: freq.get(p, 0) < min_freq)
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    return schema, freq

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("--min-freq", type=float, default=0.0)
    ap.add_argument("--max-fields", type=int, default=100, help="cap fields per class (keeps the most common); lower it if requests are too large")
    ap.add_argument("--splits", default="train", help="which split rows to learn schemas from (default train only: no test leakage)")
    a = ap.parse_args()
    from datasets import load_dataset
    sp = pd.read_csv(Path(a.cache) / "split.csv")
    ds = load_dataset("getomni-ai/ocr-benchmark", split="test").select_columns(["id", "json_schema"])
    Path("schemas").mkdir(exist_ok=True); rows = []
    use = sp[sp.split.isin(a.splits.split(","))]
    for cls, g in use.groupby("label"):
        if cls == "Unknown": continue
        schema, freq = build([ds[int(i)]["json_schema"] for i in g["id"]], a.min_freq, a.max_fields)
        (Path("schemas") / f"{cls}.json").write_text(json.dumps(schema, indent=2))
        rows += [dict(cls=cls, path=p, share=round(s, 3), n_docs=len(g)) for p, s in sorted(freq.items())]
        n_top = sum(1 for p in freq if "." not in p and "[]" not in p)
        print(f"{cls:28s} docs {len(g):3d} | fields in all docs {len(freq):4d} | kept {n_kept(schema):4d} | top-level {n_top:3d} | in >=50% of docs: {sum(s >= .5 for s in freq.values())}")
    pd.DataFrame(rows).to_csv("schemas/_report.csv", index=False)
    print("wrote schemas/*.json and schemas/_report.csv")