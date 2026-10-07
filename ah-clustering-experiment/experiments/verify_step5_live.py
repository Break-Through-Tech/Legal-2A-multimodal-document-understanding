"""Real pretrained recovery probe on the selected filesystem; no fake encoder.

The controller runs direct uninterrupted inference as its reference, terminates
one extraction child before the second checkpoint rename, then starts a fresh
child. A final cache-only child rejects any attempt to load an encoder.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from src.artifacts import read_artifact
from src.dataset import configure_cache
from src.embeddings import read_embeddings
from src.embeddings.pipeline import extract_embeddings, load_documents
from src.provenance import source_identity


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["layoutlmv3-image", "dinov3"], default="layoutlmv3-image")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--storage-root", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True, help="New filename-safe label for this probe")
    parser.add_argument("--child", choices=["crash", "resume", "cache"])
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Use only letters, digits, hyphens and underscores for run-id")
    root = args.storage_root.resolve()
    configure_cache(root)
    logs = root / "outputs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = logs / ("step5-live-" + args.run_id)
    trace = prefix.with_suffix(".trace.jsonl")
    settings = json.loads((ROOT / "configs/models" / (args.model + ".json")).read_text())
    settings["batch_size"] = 1
    dataset_dir = root / "outputs/metadata/dataset-9c71de8a0c0e5e4aab1d"
    ids = [0, 1, 2]
    from src.embeddings import models

    if args.child:
        original_load = models.load_encoder

        def traced_load(*positional, **kwargs):
            if args.child == "cache":
                raise AssertionError("Completed-cache read attempted model loading")
            encoder = original_load(*positional, **kwargs)
            original_encode = encoder.encode

            def traced_encode(images, document_ids, **options):
                result = original_encode(images, document_ids, **options)
                with trace.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"phase": args.child, "ids": document_ids}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                return result

            encoder.encode = traced_encode
            return encoder

        models.load_encoder = traced_load
        if args.child == "crash":
            original_replace = os.replace

            def interrupted_replace(source, destination, *positional, **kwargs):
                if Path(destination).name == "batch-00001.bin":
                    os._exit(74)
                return original_replace(source, destination, *positional, **kwargs)

            os.replace = interrupted_replace
        result = extract_embeddings(settings, root, dataset_dir, source_root=ROOT,
                                    document_ids=ids, device=args.device)
        save(prefix.with_suffix("." + args.child + ".json"), result)
        return

    if trace.exists() or prefix.with_suffix(".json").exists():
        raise ValueError("Use a fresh run-id; existing evidence is immutable")
    source_before = source_identity(ROOT)["source_digest"]
    _, documents = load_documents(root, dataset_dir, ids)
    baseline = {}
    started = time.perf_counter()
    encoder = models.load_encoder(settings, cache_dir=root / "cache/huggingface/hub", device=args.device)
    for row in documents:
        with Image.open(root / row["image_path"]) as original:
            with original.convert("RGB") as image:
                output = encoder.encode([image], [row["document_id"]])[0]
        baseline[row["document_id"]] = output
    baseline_seconds = time.perf_counter() - started
    del encoder
    import torch
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    command = [sys.executable, str(Path(__file__).resolve()), "--model", args.model,
               "--device", args.device, "--storage-root", str(root), "--run-id", args.run_id]
    crash = subprocess.run(command + ["--child", "crash"], check=False)
    if crash.returncode != 74:
        raise AssertionError(f"Expected injected exit 74, got {crash.returncode}")
    for phase in ("resume", "cache"):
        subprocess.run(command + ["--child", phase], check=True)
    result = json.loads(prefix.with_suffix(".resume.json").read_text())
    reused = json.loads(prefix.with_suffix(".cache.json").read_text())
    assert reused["reused"] and reused["artifact_id"] == result["artifact_id"]
    restored = read_embeddings(root, result["config"])
    assert sorted(restored) == ids
    maximum_error = 0.0
    for doc_id in ids:
        assert restored[doc_id]["token_metadata"] == baseline[doc_id]["token_metadata"]
        for name, reference in baseline[doc_id]["tensors"].items():
            actual = restored[doc_id]["tensors"][name]
            assert actual.shape == reference.shape and actual.dtype == reference.dtype
            if reference.dtype.kind in "fc":
                np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - reference), initial=0)))
            else:
                np.testing.assert_array_equal(actual, reference)
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    counts = Counter(doc_id for event in events for doc_id in event["ids"])
    assert counts == {0: 1, 1: 2, 2: 1}, counts
    assert events == [{"phase": "crash", "ids": [0]}, {"phase": "crash", "ids": [1]},
                      {"phase": "resume", "ids": [1]}, {"phase": "resume", "ids": [2]}]
    source_after = source_identity(ROOT)["source_digest"]
    assert source_after == source_before == result["config"]["source_digest"], "Source changed during verification"
    bundle = read_artifact(root, result["artifact_id"])
    checkpoint = root / "outputs/checkpoints" / ("extraction-" + result["request_id"][:20])
    evidence = {"status": "passed", "model": args.model, "document_ids": ids,
                "source_digest": source_after, "artifact_id": result["artifact_id"],
                "storage_root": str(root), "device": args.device,
                "interruption_exit_code": crash.returncode, "events": events,
                "oracle": "Direct uninterrupted real pretrained encode, outside extraction/checkpoint writer",
                "allowed_mocks": "None; termination and trace instrumentation only",
                "float_rtol": 1e-5, "float_atol": 1e-5, "integer_masks": "exact",
                "maximum_absolute_error": maximum_error, "baseline_seconds": baseline_seconds,
                "payload_bytes": sum(item["bytes"] for item in bundle["manifest"]["files"].values()),
                "checkpoint_bytes": sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()),
                "measurements": bundle["manifest"]["provenance"]["measurements"],
                "completed_cache_reused_without_encoder": True,
                "limitations": "Three image-only training documents; no OCR, full extraction, or Colab mount claim"}
    save(prefix.with_suffix(".json"), evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
