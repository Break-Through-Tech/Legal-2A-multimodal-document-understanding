"""extract(pil_image, schema) -> dict(result, valid, error). Env: ANTHROPIC_API_KEY, CLAUDE_MODEL (vision-capable model id)."""
import base64, io, json, os, re
import jsonschema

def _b64(img, max_side=2000):  # downscale: API limits on image size
    img = img.convert("RGB"); img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)          # tolerate code fences / chatter around the JSON
    return json.loads(m.group(0) if m else text)

def _check(obj, schema):
    try: jsonschema.validate(obj, schema); return None
    except jsonschema.ValidationError as e: return e.message
    except jsonschema.SchemaError as e: return f"bad schema: {e.message}"

def extract(img, schema, client=None):
    import anthropic
    client = client or anthropic.Anthropic()       # reads ANTHROPIC_API_KEY
    schema = json.loads(schema) if isinstance(schema, str) else schema
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(img)}},
               {"type": "text", "text": "Extract the data from this document as JSON that conforms to this JSON schema. "
                "Use null for values that are not visible. Reply with the JSON only.\n\nSchema:\n" + json.dumps(schema)}]
    msgs, err, obj = [{"role": "user", "content": content}], "no attempt", None
    for attempt in range(2):                        # 1 try + 1 retry
        text = client.messages.create(model=os.environ["CLAUDE_MODEL"], max_tokens=8000, messages=msgs).content[0].text
        try:
            obj = _parse(text); err = _check(obj, schema)
        except json.JSONDecodeError as e:
            obj, err = None, f"not valid JSON: {e}"
        if err is None: break
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"That output failed validation: {err}\nReturn corrected JSON only."}]
    return dict(result=obj, valid=err is None, error=err)

def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from _leaves(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from _leaves(v, f"{path}[{i}]")
    else: yield path, x

def _norm(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool): return round(float(v), 4)
    return re.sub(r"\s+", " ", str(v)).strip().lower() if v is not None else None

def field_accuracy(pred, truth):
    """Share of ground-truth leaf values (by path) that the prediction reproduces exactly (after trimming/case/number normalisation)."""
    t, p = {k: _norm(v) for k, v in _leaves(truth)}, {k: _norm(v) for k, v in _leaves(pred)} if pred is not None else {}
    return sum(p.get(k) == v for k, v in t.items()) / len(t) if t else float("nan")

if __name__ == "__main__":  # offline self-check of parsing/validation/scoring (no API call)
    s = {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}
    assert _check(_parse('```json\n{"a": 1}\n```'), s) is None and _check({"a": "x"}, s)
    assert field_accuracy({"a": [1, "X "]}, {"a": [1.0, "x"]}) == 1 and field_accuracy({"a": [1, 2]}, {"a": [1, 3]}) == 0.5
    print("extract.py self-check OK")
