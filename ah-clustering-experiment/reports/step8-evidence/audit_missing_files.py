"""Streamlit failure-state audit against an isolated copy of real saved results."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def main():
    artifact = "metrics-da8f99af889c4ea9e822"
    config = json.loads((ROOT / "outputs/metadata" / artifact / "config.json").read_text())
    checks = []
    with tempfile.TemporaryDirectory(prefix="step8-ui-negative-", dir=ROOT / "cache") as directory:
        fixture = Path(directory) / "ah-clustering-experiment"
        for identity in (artifact, config["clustering_artifact"]):
            metadata = ROOT / "outputs/metadata" / identity
            shutil.copytree(metadata, fixture / "outputs/metadata" / identity)
            manifest = json.loads((metadata / "artifact.json").read_text())
            for entry in manifest["files"].values():
                target = fixture / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / entry["path"], target)
        (fixture / "configs").mkdir()
        (fixture / "configs/evaluation.json").write_text(json.dumps({"runs": [{"model": "layoutlmv3-image", "artifact_id": config["clustering_artifact"]}]}))
        os.environ["AH_CLUSTERING_STORAGE_ROOT"] = str(fixture)
        app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=90).run()
        assert not app.exception and not app.error
        checks.append("copied real run renders overview")
        app.radio(key="page").set_value("Embedding explorer").run()
        assert not app.exception
        assert any("Saved projection unavailable" in x.value for x in app.warning)
        checks.append("missing projection displayed warning without exception")
        app.radio(key="page").set_value("Cluster explorer").run()
        assert not app.exception
        assert any("Original image unavailable" in x.value for x in app.warning)
        checks.append("missing original image displayed warning with cluster still usable")
        scores = fixture / "outputs/metrics" / artifact / "scores.json"
        scores.write_bytes(scores.read_bytes() + b" ")
        app.button[0].click().run()
        assert not app.exception
        assert any(x.value == "No completed evaluations" for x in app.title)
        assert any("invalid" in x.value.status.tolist() for x in app.dataframe)
        checks.append("corrupt score file hidden after refresh with invalid status")
        os.environ.pop("AH_CLUSTERING_STORAGE_ROOT")
    evidence = {"status": "pass", "surface": "Streamlit AppTest Python behavior", "negative_controls": checks,
                "dashboard_file_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted((ROOT / "dashboard").glob("*.py"))}}
    (ROOT / "reports/step8-evidence/independent-missing-files.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
