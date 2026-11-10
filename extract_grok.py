"""Extraction with Grok (xAI API, OpenAI-compatible) + jsonschema validation + field-level scoring.
Env vars:  XAI_API_KEY (required), GROK_MODEL (required, a vision-capable Grok model id from your xAI console),
           XAI_BASE_URL (optional, default https://api.x.ai/v1)
"""
import os, io, re, json, base64
from jsonschema import Draft7Validator

SYSTEM = ("You extract data from a document image into JSON. Return ONLY one JSON object that follows the given schema. "
          "Use only fields in the schema. If a field is not visible in the document, use null. Never invent values.")

def make_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["XAI_API_KEY"], base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"))

def image_to_data_url(img, max_side=2000):
    img = img.convert("RGB")
    if max(img.size) > max_side:
        r = max_side / max(img.size); img = img.resize((int(img.width * r), int(img.height * r)))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def parse_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m: text = m.group(1).strip()
    try: return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        return json.loads(text[i:j + 1])           # raises if still not JSON

def filled_ratio(obj):
    leaves = list(flatten(obj).values())
    return (sum(v not in (None, "", [], {}) for v in leaves) / len(leaves)) if leaves else 0.0

def extract(img, schema, client=None, model=None, retries=1):
    client = client or make_client(); model = model or os.environ["GROK_MODEL"]
    url = image_to_data_url(img); validator = Draft7Validator(schema)
    user = [{"type": "text", "text": "Schema:\n" + json.dumps(schema)}, {"type": "image_url", "image_url": {"url": url, "detail": "high"}}]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    data, errors, raw = None, [], ""
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0, response_format={"type": "json_object"})
        raw = resp.choices[0].message.content or ""
        try:
            data = parse_json(raw); errors = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in validator.iter_errors(data)]
        except Exception as e:
            data, errors = None, [f"not valid JSON: {e}"]
        if not errors: break
        msgs += [{"role": "assistant", "content": raw}, {"role": "user", "content": "That failed validation:\n" + "\n".join(errors[:10]) + "\nReturn corrected JSON only."}]
    return dict(data=data, valid=not errors, errors=errors, raw=raw, filled=filled_ratio(data) if data is not None else 0.0)

# ---------- scoring against the dataset's answer key ----------
def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items(): out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj): out.update(flatten(v, f"{prefix}[{i}]"))
    else: out[prefix] = obj
    return out

def norm(v):
    if v is None: return None
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return float(v)
    s = re.sub(r"\s+", " ", str(v)).strip().lower()
    try: return float(s.replace(",", "").replace("$", ""))
    except ValueError: return s

def field_accuracy(pred, truth):
    """Share of the answer key's leaf values that the prediction reproduces (same path, same normalized value)."""
    t = {k: norm(v) for k, v in flatten(truth).items() if v not in (None, "", [], {})}
    if not t: return None
    p = {k: norm(v) for k, v in flatten(pred or {}).items()}
    return sum(p.get(k) == v for k, v in t.items()) / len(t)
