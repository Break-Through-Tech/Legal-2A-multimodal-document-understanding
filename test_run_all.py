"""Offline check of run_all with a FAKE LLM client (no network calls to any LLM). Usage: python test_run_all.py"""
import json, re, os, tempfile
from types import SimpleNamespace as NS
import pandas as pd, run_all

class FakeClient:
    calls = 0
    def __init__(self): self.chat = NS(completions=NS(create=self.create))
    def create(self, **kw):
        FakeClient.calls += 1
        schema = json.loads(kw["messages"][1]["content"][0]["text"].removeprefix("Schema:\n"))
        return NS(choices=[NS(message=NS(content=json.dumps({k: "x" for k in schema["properties"]})))])

run_all.make_client = FakeClient
out = os.path.join(tempfile.mkdtemp(), "r.csv")
run_all.main(["--limit", "4", "--out", out]); n1 = FakeClient.calls
r = pd.read_csv(out); assert len(r) == 4
run_all.main(["--limit", "6", "--out", out]); r = pd.read_csv(out)     # resume: 4 skipped, 6 new -> 10 rows total? no: limit applies to new ones
assert len(r) == 10 and r.id.is_unique, r.id.tolist()
assert {"review_reason", "filled", "json_valid"} <= set(r.columns)
print("fake-client test OK, fake LLM calls:", FakeClient.calls)

# empty extraction (all nulls) must flag review with reason "extraction empty"
class NullClient(FakeClient):
    def create(self, **kw):
        schema = json.loads(kw["messages"][1]["content"][0]["text"].removeprefix("Schema:\n"))
        return NS(choices=[NS(message=NS(content=json.dumps({k: None for k in schema["properties"]})))])
run_all.make_client = NullClient; out2 = os.path.join(tempfile.mkdtemp(), "r.csv")
run_all.main(["--limit", "2", "--out", out2]); r2 = pd.read_csv(out2)
assert "extraction empty" in set(r2.review_reason), r2
print("empty-extraction rule OK")
