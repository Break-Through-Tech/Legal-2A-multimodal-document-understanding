"""Real checkpoint differential oracle. No mocked inference or synthetic model weights.

Run after extraction dependencies and checkpoint access are available. Each run
loads one family at a time, using prepared real image IDs from the manifest.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from src.artifacts import read_artifact
from src.embeddings import read_embeddings
from src.embeddings.pipeline import extract_embeddings
from src.provenance import source_identity


def equal(actual, expected, *, tolerance):
    """Shape and dtype are exact; arithmetic tolerance applies only to floats."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert np.isfinite(actual).all()
    if actual.dtype.kind == "f":
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    else:
        np.testing.assert_array_equal(actual, expected)


def must_detect(actual, expected, *, tolerance):
    try:
        equal(actual, expected, tolerance=tolerance)
    except AssertionError:
        return
    raise AssertionError("Negative control was not detected")


def reference_outputs(settings, documents, images, ocr, cache_dir, device):
    """Use public upstream APIs directly, never candidate preparation/collection."""
    import torch
    import transformers as tf

    family = settings["family"]
    mode = settings["input_mode"]
    model_name, processor_name = {
        "dinov3": ("DINOv3ViTModel", "AutoImageProcessor"),
        "layoutlmv3": ("LayoutLMv3Model", "LayoutLMv3Processor" if mode == "image_and_ocr" else "LayoutLMv3ImageProcessor"),
        "colpali": ("ColPaliForRetrieval", "ColPaliProcessor"),
        "colqwen2": ("ColQwen2ForRetrieval", "ColQwen2Processor"),
    }[family]
    arguments = {"revision": settings["revision"], "cache_dir": str(cache_dir)}
    processor_arguments = {**arguments, **settings["processor_options"]}
    if family == "layoutlmv3":
        processor_arguments["apply_ocr"] = False
    processor = getattr(tf, processor_name).from_pretrained(settings["checkpoint"], **processor_arguments)
    model = getattr(tf, model_name).from_pretrained(settings["checkpoint"], dtype=getattr(torch, settings["dtype"]), **arguments)
    model.to(device).eval().requires_grad_(False)
    results = {}
    with torch.inference_mode():
        for start in range(0, len(documents), settings["batch_size"]):
            rows = documents[start:start + settings["batch_size"]]
            batch_images = images[start:start + settings["batch_size"]]
            if mode == "image_and_ocr":
                words = [ocr[row["document_id"]]["words"] for row in rows]
                boxes = [ocr[row["document_id"]]["boxes"] for row in rows]
                processor.tokenizer.truncation_side = "right"
                inputs = processor(images=batch_images, text=words, boxes=boxes,
                                   truncation=True, max_length=512, padding="longest", return_tensors="pt")
                word_ids = [inputs.word_ids(batch_index=i) for i in range(len(rows))]
            else:
                inputs = processor(images=batch_images, return_tensors="pt")
                word_ids = None
            moved = {key: value.to(device=device, dtype=getattr(torch, settings["dtype"]))
                     if value.is_floating_point() else value.to(device) for key, value in inputs.items()}
            kwargs = {"use_cache": False} if family in {"colpali", "colqwen2"} else {}
            output = model(**moved, return_dict=True, **kwargs)
            native = (output.embeddings if family in {"colpali", "colqwen2"} else output.last_hidden_state).cpu().numpy()
            assert np.isfinite(native).all()
            for i, row in enumerate(rows):
                if family == "dinov3":
                    registers = model.config.num_register_tokens
                    patches = inputs["pixel_values"].shape[-1] * inputs["pixel_values"].shape[-2] // model.config.patch_size ** 2
                    assert native.shape[1] == 1 + registers + patches
                    tensors = {"cls": native[i, 0:1], "registers": native[i, 1:1 + registers], "patches": native[i, 1 + registers:]}
                elif family == "layoutlmv3":
                    visual_count = 197
                    text_count = inputs["input_ids"].shape[1] if mode == "image_and_ocr" else 0
                    assert native.shape[1] == text_count + visual_count
                    mask = np.ones(native.shape[1], dtype=bool)
                    tensors = {"cls": native[i, 0:1], "tokens": native[i], "attention_mask": mask}
                    if text_count:
                        mask[:text_count] = inputs["attention_mask"][i].numpy().astype(bool)
                        tensors.update(input_ids=inputs["input_ids"][i].numpy(), boxes=inputs["bbox"][i].numpy(),
                                       word_ids=np.array([-1 if v is None else v for v in word_ids[i]], dtype=np.int64))
                else:
                    mask = inputs["attention_mask"][i].numpy().astype(bool)
                    token_ids = inputs["input_ids"][i].numpy()
                    assert native.shape[:2] == inputs["attention_mask"].shape
                    assert native.shape[-1] == 128
                    assert np.equal(native[i][~mask], 0).all()
                    image_id = processor.image_token_id
                    tensors = {"vectors": native[i], "attention_mask": mask, "input_ids": token_ids,
                               "image_mask": (token_ids == image_id) & mask,
                               "special_mask": np.isin(token_ids, processor.tokenizer.all_special_ids) & mask}
                results[row["document_id"]] = {key: value.copy() for key, value in tensors.items()}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def verify(settings, storage_root, dataset_dir, document_ids, device):
    import torch
    start_digest = source_identity(SOURCE)["source_digest"]
    run = extract_embeddings(settings, storage_root, dataset_dir, source_root=SOURCE,
                             document_ids=document_ids, device=device)
    config = run["config"]
    saved = read_embeddings(storage_root, config)
    documents = config["documents"]
    assert set(saved) == set(document_ids)
    images = []
    for row in documents:
        with Image.open(storage_root / row["image_path"]) as image:
            images.append(image.convert("RGB"))
    ocr = None
    if config["ocr"]:
        bundle = read_artifact(storage_root, config["ocr"]["artifact_id"])
        assert config["ocr"]["content_sha256"] == bundle["manifest"]["manifest_sha256"]
        ocr = {}
        for row in bundle["tables"]["documents"].to_dict("records"):
            row["words"] = [str(word) for word in row["words"]]
            row["boxes"] = [[int(value) for value in box] for box in row["boxes"]]
            row["word_confidences"] = [float(value) for value in row["word_confidences"]]
            row["confidence"] = float(row["confidence"])
            ocr[row["document_id"]] = row
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    native = reference_outputs(settings, documents, images, ocr, storage_root / "cache/huggingface/hub", device)
    tolerance = 1e-5 if settings["dtype"] == "float32" else 3e-3
    shapes = {}
    for row in documents:
        ident = row["document_id"]
        assert saved[ident]["token_metadata"]["parent_document_id"] == ident
        assert set(saved[ident]["tensors"]) == set(native[ident])
        shapes[str(ident)] = {}
        for name, expected in native[ident].items():
            equal(saved[ident]["tensors"][name], expected, tolerance=tolerance)
            shapes[str(ident)][name] = list(expected.shape)
    first_id = documents[0]["document_id"]
    float_key = "vectors" if settings["family"] in {"colpali", "colqwen2"} else "cls"
    damaged = saved[first_id]["tensors"][float_key].copy()
    damaged.flat[0] += 1
    must_detect(damaged, native[first_id][float_key], tolerance=tolerance)
    negative_controls = ["native numeric value mutation rejected"]
    if "attention_mask" in native[first_id]:
        damaged = native[first_id]["attention_mask"][:-1]
        must_detect(damaged, native[first_id]["attention_mask"], tolerance=tolerance)
        damaged = native[first_id]["attention_mask"].copy()
        damaged[0] = ~damaged[0]
        must_detect(damaged, native[first_id]["attention_mask"], tolerance=tolerance)
        negative_controls += ["short output mask rejected", "flipped output mask rejected"]
    # Boundary mock proves cache reading performs no checkpoint inference.
    with patch("src.embeddings.models.load_encoder", side_effect=AssertionError("cache attempted model loading")):
        cached = extract_embeddings(settings, storage_root, dataset_dir, source_root=SOURCE,
                                    document_ids=list(reversed(document_ids)), device=device)
    assert cached["reused"] and cached["artifact_id"] == run["artifact_id"]
    from src.embeddings.models import load_encoder
    encoder = load_encoder(settings, cache_dir=storage_root / "cache/huggingface/hub", device=device)
    assert not encoder.model.training
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())
    observed_forwards = []
    def check_inference(module, arguments):
        assert not module.training, "Encoder entered training mode"
        assert not torch.is_grad_enabled(), "Gradient tracking enabled in model forward"
        assert torch.is_inference_mode_enabled(), "Model forward is outside inference mode"
        observed_forwards.append(True)
    handle = encoder.model.register_forward_pre_hook(check_inference)
    metamorphic = []
    ids = [row["document_id"] for row in documents]
    candidate_ocr = None
    if ocr is not None:
        candidate_ocr = []
        for ident in ids:
            record = copy.deepcopy(ocr[ident])
            record["word_confidences"] = [float(value) for value in record["word_confidences"]]
            record["boxes"] = [[int(value) for value in box] for box in record["boxes"]]
            candidate_ocr.append(record)
    if settings["family"] == "layoutlmv3" and settings["input_mode"] == "image_only":
        # Deliberately unusable OCR must have no influence on this image-only path.
        candidate_ocr = [{"words": "CORRUPTED OCR"}]
        metamorphic.append("changing OCR leaves every image-only output unchanged")
    repeated = encoder.encode(images, ids, ocr_records=candidate_ocr)
    for actual in repeated:
        for name, expected in native[actual["document_id"]].items():
            equal(actual["tensors"][name], expected, tolerance=tolerance)
    edge_cases = []
    if settings["input_mode"] == "image_and_ocr":
        nonempty = next((i for i, record in enumerate(candidate_ocr) if record["words"]), None)
        assert nonempty is not None, "Truncation edge check requires at least one actual recognized OCR word"
        parent = candidate_ocr[nonempty]
        for name in ("constructed_empty_ocr", "constructed_overlong_actual_word_replay"):
            record = copy.deepcopy(parent)
            if name == "constructed_empty_ocr":
                record.update(words=[], boxes=[], word_confidences=[], confidence=0.0, status="empty")
            else:
                count = len(parent["words"])
                record.update(words=[parent["words"][i % count] for i in range(600)],
                              boxes=[parent["boxes"][i % count] for i in range(600)],
                              word_confidences=[parent["word_confidences"][i % count] for i in range(600)], status="ok")
                record["confidence"] = float(sum(record["word_confidences"]) / 600)
            output = encoder.encode([images[nonempty]], [parent["document_id"]], ocr_records=[record])[0]
            metadata = output["token_metadata"]
            assert metadata["parent_document_id"] == parent["document_id"]
            assert metadata["truncated"] == (name == "constructed_overlong_actual_word_replay")
            expected_count = 512 if metadata["truncated"] else 2
            assert output["tensors"]["input_ids"].shape == (expected_count,)
            assert output["tensors"]["tokens"].shape[0] == expected_count + 197
            edge_cases.append((name, nonempty, record, output))
    assert observed_forwards
    assert all(parameter.grad is None for parameter in encoder.model.parameters())
    handle.remove()
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    edge_reports = []
    for name, index, record, output in edge_cases:
        expected = reference_outputs(settings, [documents[index]], [images[index]],
                                     {record["document_id"]: record}, storage_root / "cache/huggingface/hub", device)
        for key, tensor in expected[record["document_id"]].items():
            equal(output["tensors"][key], tensor, tolerance=tolerance)
        edge_reports.append({"case": name, "parent_document_id": record["document_id"],
                             "text_tokens": len(output["tensors"]["input_ids"]),
                             "total_tokens": len(output["tensors"]["tokens"]),
                             "input_authority": "constructed processor boundary input derived from actual OCR; not an OCR engine accuracy claim"})
    for image in images:
        image.close()
    assert source_identity(SOURCE)["source_digest"] == start_digest, "Source changed while oracle was running"
    return {"status": "passed", "evidence_grade": "LIVE_VERIFIED", "settings": settings,
            "source_digest": start_digest, "artifact_id": run["artifact_id"], "document_ids": document_ids,
            "shapes": shapes, "differential_tolerance": tolerance, "negative_controls": negative_controls,
            "metamorphic_checks": metamorphic, "cache_reuse_without_loading": True,
            "frozen_eval_inference_mode_verified": True, "observed_model_forwards": len(observed_forwards),
            "constructed_ocr_edge_checks": edge_reports,
            "dependencies": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "numpy", "Pillow"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, default=SOURCE)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, help="Override batch size for a real padding/batching pilot")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    report = verify(settings, args.storage_root.resolve(), args.dataset_dir.resolve(), args.ids, args.device)
    text = json.dumps(report, indent=2)
    if args.report:
        destination = args.report.resolve()
        assert destination.is_relative_to(args.storage_root.resolve()), "Report must stay inside experiment storage"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
