"""Pooling utilities for vision and document retrieval embeddings."""
from __future__ import annotations

import numpy as np


def pool_tokens(vectors: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean-pools a 2D [tokens, dim] or 3D [batch, tokens, dim] array over valid tokens,

    followed by L2 normalization.
    """
    if not isinstance(vectors, np.ndarray):
        raise ValueError("vectors must be a numpy ndarray")
    if vectors.ndim == 2:
        if mask is not None:
            if mask.ndim != 1 or len(mask) != len(vectors):
                raise ValueError("mask shape mismatch with vectors")
            selected = vectors[mask.astype(bool)]
            if len(selected) == 0:
                raise ValueError("mask selected zero tokens")
        else:
            selected = vectors
        mean_vec = np.mean(selected, axis=0, dtype=np.float64)
        norm = np.linalg.norm(mean_vec)
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("Zero or nonfinite vector norm after pooling")
        return (mean_vec / norm).astype(np.float32)

    elif vectors.ndim == 3:
        batch_size = vectors.shape[0]
        pooled = []
        for i in range(batch_size):
            m = mask[i] if mask is not None else None
            pooled.append(pool_tokens(vectors[i], m))
        return np.stack(pooled, axis=0)

    else:
        raise ValueError(f"Expected 2D or 3D vector array, got ndim={vectors.ndim}")


def pool_colpali_tokens(document: dict) -> np.ndarray:
    """Extracts and mean-pools attended image patch tokens from a ColPali document tensor dict.

    Uses `attention_mask & image_mask` to exclude prompt and non-image tokens.
    """
    if not isinstance(document, dict) or "tensors" not in document:
        raise ValueError("document must be a dict containing 'tensors'")
    tensors = document["tensors"]
    vectors = tensors.get("vectors")
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("Missing or invalid 'vectors' in document tensors")

    attn_mask = tensors.get("attention_mask")
    img_mask = tensors.get("image_mask")

    if attn_mask is None or img_mask is None:
        selected_mask = np.ones(len(vectors), dtype=bool)
    else:
        selected_mask = attn_mask.astype(bool) & img_mask.astype(bool)

    return pool_tokens(vectors, mask=selected_mask)
