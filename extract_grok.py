"""Extraction with a vision LLM through an OpenAI-compatible API (default: Groq) + jsonschema validation + field scoring.
Set in .env:  LLM_API_KEY=gsk_...   LLM_MODEL=<vision model id>   LLM_BASE_URL (optional, default Groq)
(Old names XAI_API_KEY / GROQ_API_KEY / GROK_MODEL / XAI_BASE_URL still work.)
"""
import os, io, re, json, base64, time
from dotenv import load_dotenv
load_dotenv()
from jsonschema import Draft7Validator

SYSTEM = ("You extract data from a document image into JSON. Return ONLY one JSON object that follows the given schema. "
          "Use only fields in the schema. If a field is not visible in the document, use null. Never invent values.")

def make_client():
    from openai import OpenAI
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ["XAI_API_KEY"]
    base = os.environ.get("LLM_BASE_URL") or os.environ.get("XAI_BASE_URL") or "https://api.groq.com/openai/v1"
    return OpenAI(api_key=key, base_url=base)

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

def compact_schema(node, keep_desc=True):
    """Smaller copy of the schema for the PROMPT only (validation still uses the full schema)."""
    if isinstance(node, list): return [compact_schema(x, keep_desc) for x in node]
    if not isinstance(node, dict): return node
    out = {}
    for k, v in node.items():
        if k in ("$schema", "additionalProperties") or (k == "required" and not v): continue
        if k == "description" and not keep_desc: continue
        if k == "description": out[k] = str(v)[:80]; continue
        if k == "type":
            ts = [t for t in ([v] if isinstance(v, str) else v) if t != "null"]
            out[k] = ts[0] if len(ts) == 1 else ts; continue
        if k == "enum": out[k] = [x for x in v if x is not None]; continue
        out[k] = compact_schema(v, keep_desc) if k in ("properties", "items") else v
        if k == "properties": out[k] = {p: compact_schema(s, keep_desc) for p, s in v.items()}
    return out

# (keep descriptions?, longest image side). We start at the first level and move down only if the provider says "request too large".
LEVELS = [(True, 1600), (False, 1400), (False, 1000), (False, 700)]

def too_large(e):
    s = str(e).lower()
    return "413" in s or "request too large" in s

def call_llm(client, **kw):
    """Chat call that waits and retries when the provider says 'rate limit' (429)."""
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kw)
        except Exception as e:
            if "429" not in str(e) and "RateLimit" not in type(e).__name__: raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s|m)", str(e))
            wait = (float(m.group(1)) * {"ms": .001, "s": 1, "m": 60}[m.group(2)] + 1) if m else 20 * (attempt + 1)
            print(f"  rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/6)"); time.sleep(min(wait, 90))
    raise RuntimeError("still rate limited after 6 tries")

def extract(img, schema, client=None, model=None, retries=1):
    client = client or make_client(); model = model or os.environ.get("LLM_MODEL") or os.environ["GROK_MODEL"]
    validator = Draft7Validator(schema)
    level = 0
    def base_msgs():
        keep, side = LEVELS[level]
        text = "Every field may be null if it is not visible. Schema:\n" + json.dumps(compact_schema(schema, keep), separators=(",", ":"))
        user = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image_to_data_url(img, side), "detail": "high"}}]
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    msgs = base_msgs(); history = []
    data, errors, raw = None, [], ""
    for attempt in range(retries + 1):
        while True:
            try:
                resp = call_llm(client, model=model, messages=msgs + history, temperature=0, response_format={"type": "json_object"},
                                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", 900)))
                break
            except Exception as ex:
                if too_large(ex) and level < len(LEVELS) - 1:
                    level += 1; msgs = base_msgs(); print(f"  request too large -> shrinking (schema descriptions off={not LEVELS[level][0]}, image max side {LEVELS[level][1]}px)")
                    continue
                if too_large(ex):
                    raise RuntimeError("Request is still too large at the smallest size. Use a class schema with fewer fields "
                                       "(python build_schemas.py data/embeddings --max-fields 40) or upgrade the Groq tier.") from ex
                raise
        raw = resp.choices[0].message.content or ""
        try:
            data = parse_json(raw)
        except Exception as e:
            data, errors = None, [f"not valid JSON: {e}"]
        else:
            try:
                errors = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in validator.iter_errors(data)]
            except Exception as e:                       # the SCHEMA is broken, not the answer: retrying cannot help
                return dict(data=data, valid=False, errors=[f"SCHEMA PROBLEM: {str(e)[:200]}"], raw=raw, filled=filled_ratio(data))
        if not errors: break
        history += [{"role": "assistant", "content": raw}, {"role": "user", "content": "That failed validation:\n" + "\n".join(errors[:10]) + "\nReturn corrected JSON only."}]
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


def value_recall(pred, truth):
    """Order-free score: share of the answer key's values that appear ANYWHERE in the prediction (ignores field names and list positions).
    If this is much higher than field_accuracy, the model found the values but put them in the wrong place/position."""
    t = [norm(v) for v in flatten(truth).values() if v not in (None, "", [], {})]
    if not t: return None
    bag = [norm(v) for v in flatten(pred or {}).values() if v not in (None, "", [], {})]
    hit = 0
    for v in t:
        if v in bag: bag.remove(v); hit += 1
    return hit / len(t)