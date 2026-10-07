"""Run Step 6 against completed caches. Model inference is disabled."""

from __future__ import annotations

import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class NoEncoderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src.embeddings.models" or fullname.split(".")[0] in {"transformers", "colpali_engine"}:
            raise ImportError("Clustering must consume saved embeddings; encoder loading is disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/clustering.json")
    parser.add_argument("--models", nargs="+", help="Subset of configured model names")
    parser.add_argument("--cohort", choices=["train", "val", "test", "all"])
    parser.add_argument("--similarity-device", default="cpu", help="cpu or cuda:0 for saved-token arithmetic, no model inference")
    args = parser.parse_args()
    sys.meta_path.insert(0, NoEncoderImports())
    from src.clustering.experiment import run_model
    from src.dataset import _atomic_json, configure_cache, owned_path, validate_storage_root
    from src.provenance import source_identity
    root = validate_storage_root(args.storage_root)
    configure_cache(root)
    settings = json.loads(args.config.read_text())
    if args.cohort:
        settings["cohort"] = args.cohort
    names = args.models or list(settings["models"])
    if len(names) != len(set(names)) or not set(names).issubset(settings["models"]):
        parser.error("Select unique configured model names")
    datasets = list((root / "outputs/metadata").glob("dataset-*/dataset.json"))
    if len(datasets) != 1:
        parser.error("Exactly one prepared dataset is required")
    source = source_identity(ROOT)
    started = time.perf_counter()
    results = []
    for name in names:
        try:
            results.extend(run_model(root, ROOT, datasets[0].parent, name, settings["models"][name],
                                     settings, device=args.similarity_device))
        except Exception as error:
            failure = {"model": name, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
            results.append(failure)
            print(json.dumps(failure), flush=True)
    result = {"source_digest": source["source_digest"], "settings": settings,
              "cohort": settings["cohort"], "encoder_imports_blocked": True,
              "results": results, "elapsed_seconds": time.perf_counter()-started,
              "pending": {"layoutlmv3-ocr": "Full expected-cohort embedding cache is unavailable"}
              if "layoutlmv3-ocr" not in settings["models"] else {}}
    if source_identity(ROOT)["source_digest"] != source["source_digest"]:
        result["source_changed_during_run"] = True
    report = owned_path(root, f"outputs/logs/step6-{time.time_ns()}.json")
    _atomic_json(report, result)
    print(json.dumps({"report": str(report), "completed": sum(r["status"] == "complete" for r in results),
                      "failed": sum(r["status"] == "failed" for r in results)}), flush=True)
    if result.get("source_changed_during_run") or any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
