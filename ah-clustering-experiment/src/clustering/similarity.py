"""Image-token mean-MaxSim over persisted native vectors; never loads encoders.

The symmetric distance is a dissimilarity, not a metric: distinct token sets can
have zero distance, and no triangle inequality is asserted.
"""

from __future__ import annotations

import numpy as np

from ..contracts import document_ids, require


DEFINITION = {
    "name": "image_token_mean_maxsim_v1",
    "selection": "attention_mask & image_mask",
    "special_token_policy": "Include image tokens even when special_mask is true; image_mask excludes prompt and non-image special tokens.",
    "normalization": "Per-token L2 normalization in float32; zero-norm or nonfinite selected vectors fail.",
    "directed": "mean_i max_j dot(A_i, B_j)",
    "symmetrization": "(directed + directed.T) / 2",
    "distance": "1 - similarity; zero diagonal; nonmetric dissimilarity",
    "affinity": "(similarity + 1) / 2",
    "nonmetric": True,
    "length_policy": "Mean removes source-token-count scaling; whole-source duplication and target-token duplication are invariant, but target diversity and token distributions still affect scores.",
    "precision": "float32; CUDA TF32 disabled; score range clipped only for rounding",
    "source_token_chunk": 1024,
    "target_token_chunk": 1024,
    "target_document_batch": 8,
}


def _normalized(value: np.ndarray) -> np.ndarray:
    require(isinstance(value, np.ndarray) and value.ndim == 2 and all(value.shape),
            "Expected a nonempty token-by-feature vector array")
    require(value.dtype.kind == "f" and np.isfinite(value).all(), "Vectors must be finite floating point values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = value.astype(np.float32, copy=True)
    require(np.isfinite(result).all(), "Vectors are not representable in float32")
    # Scaling first avoids overflow/underflow in the float32 norm calculation.
    scale = np.max(np.abs(result), axis=1, keepdims=True)
    require((scale > 0).all(), "Selected vectors must have nonzero L2 norm")
    result /= scale
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return np.ascontiguousarray(result)


def select_vectors(document: dict) -> np.ndarray:
    """Select attended image outputs, including image placeholders marked special."""
    require(isinstance(document, dict) and isinstance(document.get("tensors"), dict), "Missing saved document tensors")
    tensors = document["tensors"]
    vectors = tensors.get("vectors")
    require(isinstance(vectors, np.ndarray) and vectors.ndim == 2, "Missing native retrieval vectors")
    masks = {}
    for name in ("attention_mask", "image_mask", "special_mask"):
        value = tensors.get(name)
        require(isinstance(value, np.ndarray) and value.dtype == np.dtype(bool)
                and value.shape == (len(vectors),), f"Invalid or misaligned {name}")
        masks[name] = value
    selected = masks["attention_mask"] & masks["image_mask"]
    require(selected.any(), "Document has no attended image vectors")
    return _normalized(vectors[selected])


def _numpy_pair_block(source, targets):
    lengths = np.array([len(target) for target in targets])
    padded = np.zeros((len(targets), int(lengths.max()), source.shape[1]), dtype=np.float32)
    for index, target in enumerate(targets):
        padded[index, :len(target)] = target
    source_max = np.full((len(targets), len(source)), -np.inf, dtype=np.float32)
    target_max = np.full(padded.shape[:2], -np.inf, dtype=np.float32)
    for a in range(0, len(source), DEFINITION["source_token_chunk"]):
        av = source[a:a + DEFINITION["source_token_chunk"]]
        for b in range(0, padded.shape[1], DEFINITION["target_token_chunk"]):
            bv = padded[:, b:b + DEFINITION["target_token_chunk"]]
            scores = np.matmul(av, bv.swapaxes(1, 2))
            valid = np.arange(b, b + bv.shape[1])[None, :] < lengths[:, None]
            scores = np.where(valid[:, None, :], scores, -np.inf)
            source_max[:, a:a + len(av)] = np.maximum(source_max[:, a:a + len(av)], scores.max(axis=2))
            target_max[:, b:b + bv.shape[1]] = np.maximum(target_max[:, b:b + bv.shape[1]], scores.max(axis=1))
    forward = source_max.mean(axis=1)
    reverse = np.array([target_max[i, :length].mean() for i, length in enumerate(lengths)], dtype=np.float32)
    return forward, reverse


def _torch_pair_block(source, targets, device, torch):
    lengths = [len(target) for target in targets]
    padded = np.zeros((len(targets), max(lengths), source.shape[1]), dtype=np.float32)
    for index, target in enumerate(targets):
        padded[index, :len(target)] = target
    av_all = torch.from_numpy(source).to(device)
    bv_all = torch.from_numpy(padded).to(device)
    sizes = torch.tensor(lengths, device=device)
    source_max = torch.full((len(targets), len(source)), -torch.inf, device=device)
    target_max = torch.full(padded.shape[:2], -torch.inf, device=device)
    for a in range(0, len(source), DEFINITION["source_token_chunk"]):
        av = av_all[a:a + DEFINITION["source_token_chunk"]]
        for b in range(0, padded.shape[1], DEFINITION["target_token_chunk"]):
            bv = bv_all[:, b:b + DEFINITION["target_token_chunk"]]
            scores = torch.matmul(av, bv.transpose(1, 2))
            valid = torch.arange(b, b + bv.shape[1], device=device)[None, :] < sizes[:, None]
            scores.masked_fill_(~valid[:, None, :], -torch.inf)
            source_max[:, a:a + len(av)] = torch.maximum(source_max[:, a:a + len(av)], scores.amax(dim=2))
            target_max[:, b:b + bv.shape[1]] = torch.maximum(target_max[:, b:b + bv.shape[1]], scores.amax(dim=1))
    return (source_max.mean(dim=1).cpu().numpy(),
            torch.stack([target_max[i, :length].mean() for i, length in enumerate(lengths)]).cpu().numpy())


def compute_similarity(vectors: list[np.ndarray], device="cpu", progress=None) -> dict:
    """Compute both directions once per pair with bounded score tiles.

    CPU uses NumPy. CUDA uploads at most one source and eight target documents;
    each score tile is at most 8 x 1024 x 1024 float32 values (32 MiB). The largest
    document and padded target block determine additional vector storage.
    ``progress`` receives completed_pairs, total_pairs and source_index.
    """
    require(isinstance(vectors, list) and bool(vectors), "Provide a nonempty list of document vectors")
    normalized = [_normalized(value) for value in vectors]
    require(len({value.shape[1] for value in normalized}) == 1, "Document feature dimensions differ")
    require(isinstance(device, str) and (device == "cpu" or device == "cuda" or device.startswith("cuda:")),
            "Similarity device must be cpu or cuda")
    torch = None
    previous_tf32 = None
    if device != "cpu":
        import torch
        require(torch.cuda.is_available(), "CUDA similarity requested but unavailable")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    count = len(normalized)
    directed = np.eye(count, dtype=np.float32)
    completed, total = 0, count * (count - 1) // 2
    try:
        for i, source in enumerate(normalized):
            for start in range(i + 1, count, DEFINITION["target_document_batch"]):
                stop = min(count, start + DEFINITION["target_document_batch"])
                targets = normalized[start:stop]
                if torch is None:
                    forward, reverse = _numpy_pair_block(source, targets)
                else:
                    with torch.inference_mode():
                        forward, reverse = _torch_pair_block(source, targets, device, torch)
                directed[i, start:stop] = forward
                directed[start:stop, i] = reverse
                completed += stop - start
                if progress is not None:
                    progress({"completed_pairs": completed, "total_pairs": total, "source_index": i})
    finally:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    np.clip(directed, -1, 1, out=directed)
    similarity = (directed + directed.T) * np.float32(.5)
    distance = np.float32(1) - similarity
    affinity = (similarity + np.float32(1)) * np.float32(.5)
    result = {"directed": directed, "similarity": similarity, "distance": distance, "affinity": affinity}
    validate_matrices(result, list(range(count)), list(range(count)))
    return result


def validate_matrices(matrices: dict, ids, expected_ids) -> None:
    """Reject incorrect document order, invalid score ranges and transforms."""
    actual = document_ids(ids)
    require(actual == document_ids(expected_ids), "Similarity document ID order differs from expected IDs")
    require(isinstance(matrices, dict) and set(matrices) == {"directed", "similarity", "distance", "affinity"},
            "Missing or unexpected similarity matrices")
    shape = (len(actual), len(actual))
    tolerance = 2e-6
    for name, value in matrices.items():
        require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind == "f"
                and np.isfinite(value).all(), f"Invalid {name} matrix")
        low, high = {"directed": (-1, 1), "similarity": (-1, 1), "distance": (0, 2), "affinity": (0, 1)}[name]
        require((value >= low - tolerance).all() and (value <= high + tolerance).all(), f"{name} outside defined range")
        require(np.allclose(np.diag(value), 0 if name == "distance" else 1, atol=tolerance, rtol=0),
                f"Invalid {name} diagonal")
        if name != "directed":
            require(np.allclose(value, value.T, atol=tolerance, rtol=0), f"Asymmetric {name}")
    expected_similarity = (matrices["directed"] + matrices["directed"].T) / 2
    require(np.allclose(matrices["similarity"], expected_similarity, atol=tolerance, rtol=0), "Incorrect mean symmetrization")
    require(np.allclose(matrices["distance"], 1 - matrices["similarity"], atol=tolerance, rtol=0), "Incorrect distance transform")
    require(np.allclose(matrices["affinity"], (1 + matrices["similarity"]) / 2, atol=tolerance, rtol=0), "Incorrect affinity transform")
