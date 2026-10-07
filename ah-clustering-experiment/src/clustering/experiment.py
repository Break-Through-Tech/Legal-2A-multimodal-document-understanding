"""Run label-free clustering from immutable native embedding artifacts."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .runner import fit_clusters, validate_matrix
from .similarity import DEFINITION, select_vectors, compute_similarity, validate_matrices
from ..artifacts import ArtifactWriter, artifact_identity, read_artifact
from ..contracts import clustering_config, document_ids, require
from ..dataset import (_atomic_json, file_sha256, fingerprint, owned_path, select_cohort, validate_storage_root,
                       _validate_preparation_metadata, validate_assignments as validate_dataset_assignments,
                       DATASET_NAME, DATASET_REVISION)
from ..embeddings import _decode
from ..provenance import capture_provenance, source_identity


def load_cohort(root: Path, dataset_dir: Path, name: str) -> tuple[dict, list[dict]]:
    require(dataset_dir.resolve().is_relative_to(root.resolve()), "Dataset metadata must stay inside storage root")
    metadata = json.loads((dataset_dir / "dataset.json").read_text())
    _validate_preparation_metadata(metadata)
    identity = metadata["identity"]
    require(metadata["status"] == "complete"
            and metadata["identity_sha256"] == fingerprint(identity)
            and metadata["dataset_id"] == "dataset-" + fingerprint(identity)[:20], "Invalid dataset identity")
    require(identity["dataset"] == {"name": DATASET_NAME, "revision": DATASET_REVISION, "upstream_split": "test"},
            "Unexpected dataset source")
    require(file_sha256(dataset_dir / "manifest.parquet") == metadata["manifest_sha256"], "Dataset manifest checksum differs")
    manifest = pd.read_parquet(dataset_dir / "manifest.parquet")
    validate_dataset_assignments(manifest.rename(columns={"document_id": "id"}))
    require((manifest.dataset_revision == DATASET_REVISION).all()
            and (manifest.shared_csv_sha256 == identity["shared_csv_sha256"]).all(), "Dataset row provenance differs")
    ids = sorted(select_cohort(manifest, name))
    documents = manifest.loc[manifest.document_id.isin(ids), ["document_id", "image_path", "image_sha256"]].sort_values("document_id").to_dict("records")
    return {"dataset_id": metadata["dataset_id"], "cohort": name, "document_ids": ids,
            "designation": "initial_training_partition" if name == "train" else "exploratory",
            "manifest_sha256": metadata["manifest_sha256"], "dataset_identity_sha256": metadata["identity_sha256"]}, documents


def validate_assignments(bundle: dict):
    require(bundle["manifest"]["kind"] == "clusters", "Expected a clustering artifact")
    require(set(bundle["tables"]) == {"assignments"} and not bundle["arrays"], "Unexpected cluster payloads")
    frame = bundle["tables"]["assignments"]
    require(set(frame.columns) == {"document_id", "cluster_id"}, "Unexpected assignment schema")
    ids = document_ids(frame.document_id)
    require(ids == bundle["config"]["cohort"]["document_ids"], "Assignment ID order or coverage differs")
    labels = frame.cluster_id.to_numpy()
    require(labels.dtype.kind in "iu" and (labels >= -1).all(), "Invalid cluster labels")
    summary = bundle["json"]["summary"]
    require(summary["document_count"] == len(ids) and summary["noise_count"] == int((labels == -1).sum())
            and summary["cluster_count"] == len(set(labels) - {-1}), "Assignment summary differs")


def _similarity_bundle(bundle: dict):
    ids = bundle["tables"]["ordered_ids"].sort_values("position").document_id.tolist()
    require(bundle["tables"]["ordered_ids"].position.tolist() == list(range(len(ids))), "Invalid matrix row positions")
    validate_matrices(bundle["arrays"], ids, bundle["config"]["cohort"]["document_ids"])
    require(bundle["json"]["validation"]["passed"] is True, "Similarity has no passed validation")


def similarity_artifact(root, source_root, embedding, documents, cohort, device, dependencies):
    started = time.perf_counter()
    config = {"embedding_artifact": embedding["manifest"]["artifact_id"],
              "embedding_identity": embedding["manifest"]["identity_sha256"],
              "embedding_manifest_sha256": embedding["manifest"]["manifest_sha256"],
              "cohort": cohort, "definition": DEFINITION, "device": device,
              "dependencies": {"numpy": dependencies["numpy"], "torch": importlib.metadata.version("torch") if device.startswith("cuda") else None},
              # Clustering configuration/algorithms do not change saved-token mathematics.
              # Full execution source provenance is still recorded in the manifest.
              "implementation_files_sha256": {name: file_sha256(source_root / name) for name in (
                  "src/clustering/similarity.py", "src/clustering/experiment.py", "src/artifacts.py",
                  "src/contracts.py", "src/dataset.py", "src/embeddings/__init__.py")}}
    artifact_id, identity = artifact_identity("similarities", config)
    marker = owned_path(root, f"outputs/metadata/{artifact_id}/artifact.json")
    if marker.exists():
        bundle = read_artifact(root, artifact_id, expected_config=config)
        _similarity_bundle(bundle)
        return bundle
    ids = cohort["document_ids"]
    vectors = [select_vectors(documents[doc_id]) for doc_id in ids]
    progress_path = owned_path(root, f"outputs/logs/{artifact_id}.json")
    last_report = -1

    def progress(event):
        nonlocal last_report
        percentage = int(100 * event["completed_pairs"] / event["total_pairs"])
        if percentage >= last_report + 5:
            last_report = percentage
            _atomic_json(progress_path, {"status": "running", **event, "percent": percentage})
            print(json.dumps({"similarity": artifact_id, "percent": percentage}), flush=True)

    with ArtifactWriter(root, "similarities", config) as writer:
        matrices = compute_similarity(vectors, device=device, progress=progress)
        validate_matrices(matrices, ids, ids)
        # Fresh small direct NumPy calculation checks actual saved-token scores.
        selected = sorted(set([0, len(ids) // 2, len(ids) - 1]))
        errors = []
        for i in selected:
            for j in selected:
                reference = float((vectors[i].astype(np.float64) @ vectors[j].astype(np.float64).T).max(axis=1).mean())
                errors.append(abs(reference - float(matrices["directed"][i, j])))
        require(max(errors) <= 2e-5, "Real-token MaxSim differs from direct NumPy reference")
        validation = {"passed": True, "direct_reference_max_error": max(errors), "tolerance": 2e-5,
                      "reference_document_ids": [ids[i] for i in selected],
                      "selected_token_count_min": min(map(len, vectors)), "selected_token_count_max": max(map(len, vectors)),
                      "max_directional_asymmetry": float(np.max(np.abs(matrices["directed"] - matrices["directed"].T))),
                      "similarity_min": float(matrices["similarity"].min()), "similarity_max": float(matrices["similarity"].max()),
                      "distance_is_metric": False}
        for name, value in matrices.items():
            writer.array(name, value)
        writer.table("ordered_ids", pd.DataFrame({"document_id": ids, "position": range(len(ids))}))
        writer.json("validation", validation)
        writer.complete(capture_provenance(source_root, started, validation), validator=_similarity_bundle)
    _atomic_json(progress_path, {"status": "complete", "artifact_id": artifact_id, "elapsed_seconds": time.perf_counter()-started})
    return read_artifact(root, artifact_id, expected_config=config)


def run_model(root: Path, source_root: Path, dataset_dir: Path, name: str, embedding_id: str,
              settings: dict, device: str = "cpu") -> list[dict]:
    root = validate_storage_root(root)
    cohort, images = load_cohort(root, dataset_dir, settings["cohort"])
    embedding = read_artifact(root, embedding_id)
    native = _decode(embedding)
    config = embedding["config"]
    require(config["dataset"]["dataset_id"] == cohort["dataset_id"]
            and config["dataset"]["manifest_sha256"] == cohort["manifest_sha256"]
            and config["dataset"]["identity_sha256"] == cohort["dataset_identity_sha256"], "Embedding dataset differs from cohort")
    indexed_images = {r["document_id"]: r for r in config["documents"]}
    require(all(indexed_images.get(row["document_id"]) == row for row in images), "Cohort images differ from embedding cache")
    ids = cohort["document_ids"]
    dependencies = {key: importlib.metadata.version(key) for key in ("scikit-learn", "numpy", "scipy", "threadpoolctl")}
    precomputed = "cls" not in config["tensors"]
    similarity = None
    if precomputed:
        similarity = similarity_artifact(root, source_root, embedding, native, cohort, device, dependencies)
    else:
        require(all(native[key]["tensors"]["cls"].shape[0] == 1 for key in ids), "Expected one CLS vector per document")
        features = np.vstack([native[key]["tensors"]["cls"][0] for key in ids]).astype(np.float64)
        require(settings["normalization"] in {"l2", "none"}, "Unsupported CLS normalization")
        if settings["normalization"] == "l2":
            lengths = np.linalg.norm(features, axis=1, keepdims=True)
            require((lengths > 0).all(), "Cannot normalize zero CLS vector")
            features = features / lengths
        validate_matrix(features, len(ids))
    results = []
    for algorithm, original_parameters in settings["algorithms"].items():
        if precomputed and algorithm == "kmeans":
            results.append({"model": name, "algorithm": algorithm, "status": "unavailable",
                            "reason": "Standard K-means needs fixed feature vectors; retained-token similarity is not a feature matrix."})
            continue
        parameters = dict(original_parameters)
        if precomputed:
            if algorithm == "spectral":
                parameters["affinity"] = "precomputed"
                parameters.pop("gamma", None)
            else:
                parameters["metric"] = "precomputed"
            data = similarity["arrays"]["affinity" if algorithm == "spectral" else "distance"]
        else:
            data = features
        count_policy = settings["cluster_count_policy"] if algorithm != "hdbscan" else {
            "mode": "independently_chosen", "n_clusters": None,
            "rationale": "HDBSCAN discovers density groups using predeclared min_cluster_size and min_samples; no label-derived cluster count."}
        run_config = clustering_config(
            embedding_artifact=embedding_id, embedding_identity=embedding["manifest"]["identity_sha256"], model=config["model"],
            representation="retained_image_tokens" if precomputed else "cls", token_selection={"kind": "attention_and_image" if precomputed else "cls"},
            pooling={"kind": "none"}, normalization={"kind": "per_token_l2" if precomputed else settings["normalization"]},
            similarity={"artifact_id": similarity["manifest"]["artifact_id"], "identity_sha256": similarity["manifest"]["identity_sha256"],
                        "definition": DEFINITION} if precomputed else None,
            algorithm=algorithm, parameters=parameters, cohort=cohort, seed=settings["seed"],
            cluster_count_policy=count_policy, source_digest=source_identity(source_root)["source_digest"])
        run_config.update(dependencies=dependencies, cpu_threads=1, embedding_manifest_sha256=embedding["manifest"]["manifest_sha256"])
        run_id, _ = artifact_identity("clusters", run_config)
        log = owned_path(root, f"outputs/logs/{run_id}.json")
        started = time.perf_counter()
        try:
            marker = owned_path(root, f"outputs/metadata/{run_id}/artifact.json")
            reused = marker.exists()
            if reused:
                bundle = read_artifact(root, run_id, expected_config=run_config)
                validate_assignments(bundle)
            else:
                with ArtifactWriter(root, "clusters", run_config) as writer:
                    labels = fit_clusters(data, algorithm, parameters, settings["seed"], precomputed=precomputed)
                    repeated = fit_clusters(data, algorithm, parameters, settings["seed"], precomputed=precomputed)
                    require(adjusted_rand_score(labels, repeated) == 1.0 and np.array_equal(labels == -1, repeated == -1),
                            "Repeated fit changed partition or noise membership")
                    summary = {"document_count": len(ids), "cluster_count": len(set(labels) - {-1}),
                               "noise_count": int((labels == -1).sum()), "repeat_partition_equal": True}
                    writer.table("assignments", pd.DataFrame({"document_id": ids, "cluster_id": labels}))
                    writer.json("summary", summary)
                    writer.complete(capture_provenance(source_root, started, summary), validator=validate_assignments)
                bundle = read_artifact(root, run_id, expected_config=run_config)
            result = {"model": name, "algorithm": algorithm, "status": "complete", "artifact_id": run_id,
                      "reused": reused, **bundle["json"]["summary"], "elapsed_seconds": time.perf_counter()-started}
        except Exception as error:
            result = {"model": name, "algorithm": algorithm, "status": "failed", "artifact_id": run_id,
                      "error_type": type(error).__name__, "error": str(error)}
        _atomic_json(log, result)
        print(json.dumps(result), flush=True)
        results.append(result)
    return results
