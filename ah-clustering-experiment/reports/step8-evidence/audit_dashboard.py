"""Independent saved-file differential and disposable corruption probe."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dashboard.data import image_bytes, load_projection, load_run, scan_catalog
from dashboard.comparison import compare_runs


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    catalog = scan_catalog(ROOT)
    configured = read_json(ROOT / "configs/evaluation.json")["runs"]
    assert len(catalog["runs"]) == len(configured) == 14
    assert {r["clustering_artifact"] for r in catalog["runs"]} == {r["artifact_id"] for r in configured}
    csv = pd.read_csv(ROOT.parent / "data/split.csv")
    train = csv.loc[csv.split == "train"].set_index("id").sort_index()
    assert len(train) == 700
    records, projection_ids, checked_files = [], set(), set()
    score_slots = 0
    for descriptor in catalog["runs"]:
        artifact_id = descriptor["artifact_id"]
        run = load_run(ROOT, artifact_id)
        manifest = read_json(ROOT / "outputs/metadata" / artifact_id / "artifact.json")
        config = read_json(ROOT / "outputs/metadata" / artifact_id / "config.json")
        raw = {}
        for name, entry in manifest["files"].items():
            path = ROOT / entry["path"]
            payload = path.read_bytes()
            assert len(payload) == entry["bytes"]
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
            checked_files.add(entry["path"])
            raw[name] = pd.read_parquet(path) if entry["format"] == "parquet" else read_json(path)
        pd.testing.assert_frame_equal(run["documents"], raw["documents"])
        pd.testing.assert_frame_equal(run["clusters"], raw["clusters"])
        assert run["scores"] == raw["scores"]
        assert run["summary"] == raw["summary"]
        for scope in raw["scores"].values():
            score_slots += len(scope["metrics"])
        docs = run["documents"].set_index("document_id").sort_index()
        assert docs.index.tolist() == train.index.tolist()
        assert docs.label.tolist() == train.label.tolist()
        assert docs.split.tolist() == ["train"] * 700
        for key in ("document_count", "cluster_count", "noise_percentage"):
            assert descriptor[key] == raw["summary"][key]
        cluster_meta = read_json(ROOT / "outputs/metadata" / config["clustering_artifact"] / "artifact.json")
        cluster_config = read_json(ROOT / "outputs/metadata" / config["clustering_artifact"] / "config.json")
        assert descriptor["algorithm"] == cluster_config["algorithm"]
        assert descriptor["representation"] == cluster_config["representation"]
        assert descriptor["seed"] == cluster_config["seed"]
        assert descriptor["clustering_seconds"] == cluster_meta["provenance"]["runtime"]["elapsed_seconds"]
        assert descriptor["evaluation_seconds"] == manifest["provenance"]["runtime"]["elapsed_seconds"]
        for ref in run["projections"]:
            projection = load_projection(ROOT, ref, docs.index.tolist())
            pm = read_json(ROOT / "outputs/metadata" / ref["artifact_id"] / "artifact.json")
            coords = pd.read_parquet(ROOT / pm["files"]["coordinates"]["path"])
            pd.testing.assert_frame_equal(projection, coords)
            assert projection.document_id.tolist() == train.index.tolist()
            projection_ids.add(ref["artifact_id"])
        renamed = copy.deepcopy(run)
        renamed["documents"].loc[~renamed["documents"].is_noise, "cluster_id"] += 10000
        comparison = compare_runs(run, renamed)
        expected_nonnoise = int((docs.cluster_id != -1).sum())
        assert comparison["summary"]["compared_count"] == expected_nonnoise
        assert comparison["summary"]["agreement_count"] == expected_nonnoise
        assert comparison["summary"]["agreement_fraction"] == 1
        assert comparison["summary"]["comparison_coverage"] == expected_nonnoise / 700
        row = docs.iloc[0]
        with Image.open(ROOT / row.image_path) as original, Image.open(io.BytesIO(image_bytes(ROOT, row))) as displayed:
            assert np.array_equal(np.asarray(original.convert("RGB")), np.asarray(displayed))
        records.append({"artifact_id": artifact_id, "model": descriptor["model"], "algorithm": descriptor["algorithm"],
                        "document_count": len(docs), "noise_count": 700 - expected_nonnoise,
                        "metric_slots": sum(len(s["metrics"]) for s in raw["scores"].values()),
                        "renumbered_agreement": comparison["summary"]["agreement_fraction"]})
    assert len(projection_ids) == 6 and score_slots == 280
    first = catalog["runs"][0]
    negatives = []
    with tempfile.TemporaryDirectory(prefix="step8-independent-", dir=ROOT / "cache") as temporary:
        fixture = Path(temporary) / "ah-clustering-experiment"
        for identity in (first["artifact_id"], first["clustering_artifact"]):
            metadata = ROOT / "outputs/metadata" / identity
            shutil.copytree(metadata, fixture / "outputs/metadata" / identity)
            for entry in read_json(metadata / "artifact.json")["files"].values():
                target = fixture / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / entry["path"], target)
        (fixture / "configs").mkdir()
        (fixture / "configs/evaluation.json").write_text(json.dumps({"runs": [{"artifact_id": first["clustering_artifact"], "model": first["model"]}]}))
        assert len(scan_catalog(fixture)["runs"]) == 1
        bad = fixture / "outputs/metrics" / first["artifact_id"] / "scores.json"
        original = bad.read_bytes()
        bad.write_bytes(original + b" ")
        rejected = scan_catalog(fixture)
        assert not rejected["runs"] and any(row["status"] == "invalid" for row in rejected["issues"])
        negatives.append("checksum corruption excluded run and exposed invalid status")
        bad.unlink()
        rejected = scan_catalog(fixture)
        assert not rejected["runs"] and rejected["issues"]
        negatives.append("missing score file excluded run and exposed issue")
        try:
            image_bytes(fixture, {"image_path": "outputs/missing.png"})
        except ValueError:
            negatives.append("missing original image produced explicit ValueError")
        else:
            raise AssertionError("missing image accepted")
    evidence = {"status": "pass", "run_count": len(records), "projection_count": len(projection_ids),
                "score_slots": score_slots, "training_document_count": len(train), "checked_payload_files": len(checked_files),
                "runs": records, "issues": catalog["issues"], "negative_controls": negatives,
                "dashboard_file_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted((ROOT / "dashboard").glob("*.py"))}}
    output = ROOT / "reports/step8-evidence/independent-artifact-audit.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items() if k not in {"runs", "dashboard_file_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
