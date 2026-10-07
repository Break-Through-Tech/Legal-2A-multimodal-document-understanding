"""Native, pickle-free batch restart records. A whole inference batch is committed."""

import io
import json
import zipfile

import numpy as np

from . import _validate_document
from ..contracts import require


def encode_batch(outputs: list[dict], config: dict, ids: list[int]) -> bytes:
    require([row["document_id"] for row in outputs] == ids, "Encoder changed document IDs or batch order")
    arrays = {}
    metadata = []
    for index, row in enumerate(outputs):
        _validate_document(row, config)
        metadata.append({"document_id": row["document_id"], "token_metadata": row["token_metadata"]})
        arrays.update({f"d{index}_{name}": array for name, array in row["tensors"].items()})
    arrays["metadata"] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def decode_batch(data: bytes | None, config: dict, ids: list[int]) -> list[dict] | None:
    if data is None:
        return None
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].tobytes())
            require(isinstance(metadata, list) and [row["document_id"] for row in metadata] == ids,
                    "Checkpoint document IDs differ")
            require(set(archive.files) == {"metadata"} | {f"d{i}_{name}" for i in range(len(ids)) for name in config["tensors"]},
                    "Checkpoint tensor names differ")
            outputs = [{**row, "tensors": {name: archive[f"d{i}_{name}"] for name in config["tensors"]}}
                       for i, row in enumerate(metadata)]
            for row in outputs:
                _validate_document(row, config)
            return outputs
    except (ValueError, TypeError, KeyError, OSError, EOFError, zipfile.BadZipFile):
        return None
