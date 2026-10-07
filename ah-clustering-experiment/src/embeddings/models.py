"""Frozen native Transformers encoders. No application-level pooling or normalization.

Imports of torch/Transformers are lazy: reading stored artifacts needs neither.
See reports/MODEL_API_NOTES.md for upstream contracts and checkpoint pins.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from ..contracts import document_ids as validate_ids, require, numeric_array


def validate_settings(settings: dict) -> None:
    require(settings.get("family") in {"dinov3", "colpali", "colqwen2", "layoutlmv3"}, "Unknown encoder family")
    require(isinstance(settings.get("checkpoint"), str) and bool(settings["checkpoint"].strip()), "Missing checkpoint")
    require(re.fullmatch(r"[0-9a-f]{40}", settings.get("revision", "")) is not None, "Immutable model revision required")
    require(settings.get("dtype") in {"float32", "float16"}, "Use numpy-compatible float32 or float16")
    require(type(settings.get("batch_size")) is int and settings["batch_size"] > 0, "Positive batch_size required")
    require(settings.get("input_mode") in {"image_only", "image_and_ocr"}, "Unknown input mode")
    require(settings["input_mode"] == "image_only" or settings["family"] == "layoutlmv3", "Only LayoutLMv3 consumes OCR")
    options = settings.get("processor_options", {})
    require(isinstance(options, dict), "processor_options must be an object")
    # Options are image processor overrides, never arbitrary model/download kwargs.
    allowed = {"size", "crop_size", "do_resize", "do_center_crop", "do_rescale", "rescale_factor",
               "do_normalize", "image_mean", "image_std", "resample", "min_pixels", "max_pixels"}
    require(set(options) <= allowed, "Unsupported processor_options; only image transformations may be overridden")
    json.dumps(settings, allow_nan=False)


def validate_output_alignment(vectors: np.ndarray, mask: np.ndarray, token_ids: np.ndarray) -> None:
    """A processor mask is accepted only after checking the actual returned axes."""
    numeric_array(vectors)
    require(vectors.ndim == 3, "Expected output [batch, tokens, dimension]")
    require(mask.ndim == 2 and mask.shape == vectors.shape[:2], "Output token/mask alignment mismatch")
    require(token_ids.shape == mask.shape, "Output token/ID alignment mismatch")
    require(np.isin(mask, [0, 1]).all(), "Attention mask must be binary")
    require(token_ids.dtype.kind in "iu", "Token IDs must be integers")


def _json_config(value) -> dict:
    # Transformers configs expose JSON serializers that handle dtype and enums.
    if hasattr(value, "to_diff_dict"):
        return json.loads(value.to_json_string(use_diff=False))
    if hasattr(value, "to_json_string"):
        return json.loads(value.to_json_string())
    return json.loads(json.dumps(value.to_dict(), default=str, allow_nan=False))


def _processor_metadata(processor, options: dict) -> dict:
    image_processor = getattr(processor, "image_processor", processor)
    result = {"class": type(processor).__name__, "image_processor": _json_config(image_processor),
              "overrides": options}
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
        result["tokenizer"] = {"class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side,
                               "truncation_side": tokenizer.truncation_side,
                               "special_tokens_map": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
                               "model_max_length": tokenizer.model_max_length}
        if hasattr(tokenizer, "backend_tokenizer"):
            result["tokenizer"]["backend_sha256"] = hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        for name in ("visual_prompt_prefix", "image_seq_length", "image_token", "chat_template"):
            value = getattr(processor, name, None)
            if value is not None:
                result[name] = value
    return result


def load_encoder(settings: dict, *, cache_dir: Path, device: str):
    """Load a fully merged immutable checkpoint, its matching processor, and freeze it."""
    validate_settings(settings)
    import torch
    from transformers import (AutoImageProcessor, DINOv3ViTModel, ColPaliForRetrieval,
                              ColPaliProcessor, ColQwen2ForRetrieval, ColQwen2Processor,
                              LayoutLMv3Model, LayoutLMv3Processor, LayoutLMv3ImageProcessor)

    dtype = getattr(torch, settings["dtype"])
    family = settings["family"]
    common = {"revision": settings["revision"], "cache_dir": str(cache_dir)}
    classes = {"dinov3": (DINOv3ViTModel, AutoImageProcessor),
               "colpali": (ColPaliForRetrieval, ColPaliProcessor),
               "colqwen2": (ColQwen2ForRetrieval, ColQwen2Processor),
               "layoutlmv3": (LayoutLMv3Model, LayoutLMv3Processor)}
    model_class, processor_class = classes[family]
    if family == "layoutlmv3" and settings["input_mode"] == "image_only":
        processor_class = LayoutLMv3ImageProcessor
    processor_kwargs = {**common, **settings.get("processor_options", {})}
    if family == "layoutlmv3":
        processor_kwargs["apply_ocr"] = False
    processor = processor_class.from_pretrained(settings["checkpoint"], **processor_kwargs)
    if family == "layoutlmv3" and settings["input_mode"] == "image_and_ocr":
        require(processor.tokenizer.is_fast, "LayoutLMv3 requires a fast tokenizer for OCR word alignment")
        processor.tokenizer.truncation_side = "right"
    model = model_class.from_pretrained(settings["checkpoint"], dtype=dtype, **common)
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    return NativeEncoder(settings, model, processor, device)


class NativeEncoder:
    def __init__(self, settings, model, processor, device):
        self.settings = json.loads(json.dumps(settings))
        self.model, self.processor, self.device = model, processor, device
        self.family = settings["family"]
        self.mode = settings["input_mode"]
        self.metadata = {
            "model": {"checkpoint": settings["checkpoint"], "revision": settings["revision"],
                      "class": type(model).__name__, "config": _json_config(model.config)},
            "processor": _processor_metadata(processor, settings.get("processor_options", {})),
            "output_policy": {"pooling": "none", "additional_normalization": "none",
                              "native_retrieval_l2_normalization": self.family in {"colpali", "colqwen2"},
                              "retain_padding": True, "dtype": settings["dtype"],
                              "layout_text_max_length": 512, "layout_truncation": "first_512_subtokens_including_specials",
                              "empty_ocr": "text_special_tokens_plus_image", "image_only_ocr": "ignored"},
        }
        self.metadata["runtime"] = {"device": str(device), "batch_size": settings["batch_size"], "dtype": settings["dtype"]}
        self.metadata["transformations"] = {
            "image": {"conversion": "PIL.convert(RGB)", "processor": self.metadata["processor"]["image_processor"]},
            "word_coordinates": "actual OCR normalized integer xyxy 0..1000; unchanged by tokenizer" if self.mode == "image_and_ocr" else "unused",
        }
        self.metadata["token_policy"] = dict(self.metadata["output_policy"])
        self.tensor_schema = self._schema()

    def _schema(self):
        dtype = self.settings["dtype"]
        hidden = self.model.config.embedding_dim if self.family in {"colpali", "colqwen2"} else self.model.config.hidden_size
        def spec(kind, group, tail=None, dt=None):
            return {"dtype": dt or dtype, "tail_shape": tail if tail is not None else [hidden],
                    "role": kind, "alignment": group}
        if self.family == "dinov3":
            return {name: spec("embedding", name) for name in ("cls", "registers", "patches")}
        if self.family in {"colpali", "colqwen2"}:
            return {"vectors": spec("embedding", "retrieval"), "input_ids": spec("token_ids", "retrieval", [], "int64"),
                    **{name: spec("mask", "retrieval", [], "bool") for name in ("attention_mask", "image_mask", "special_mask")}}
        result = {"cls": spec("embedding", "cls"), "tokens": spec("embedding", "sequence"),
                  "attention_mask": spec("mask", "sequence", [], "bool")}
        if self.mode == "image_and_ocr":
            result.update(input_ids=spec("token_ids", "text", [], "int64"),
                          boxes=spec("coordinates", "text", [4], "int64"),
                          word_ids=spec("token_ids", "text", [], "int64"))
        return result

    def _numpy(self, tensor):
        array = tensor.detach().cpu().numpy().copy()
        numeric_array(array)
        if array.dtype.kind == "f":
            require(array.dtype == np.dtype(self.settings["dtype"]), "Native model output dtype differs from configured dtype")
        return array

    def encode(self, images: list[Image.Image], document_ids: list[int], ocr_records: list[dict] | None = None) -> list[dict]:
        import torch
        ids = validate_ids(document_ids)
        require(len(images) == len(ids), "Image/document count mismatch")
        require(all(isinstance(im, Image.Image) and im.width > 0 and im.height > 0 for im in images), "Expected nonempty PIL images")
        ocr = None
        if self.mode == "image_and_ocr":
            require(isinstance(ocr_records, list) and len(ocr_records) == len(ids), "Actual OCR records required")
            require(validate_ids(r["document_id"] for r in ocr_records) == ids, "OCR/document ID order mismatch")
            for record in ocr_records:
                self._validate_ocr(record)
            ocr = ocr_records
        result = []
        size = self.settings["batch_size"]
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(ids), size):
                batch_images = [im.convert("RGB") for im in images[start:start + size]]
                batch_ocr = None if ocr is None else ocr[start:start + size]
                inputs, alignment = self._prepare(batch_images, batch_ocr)
                moved = {k: v.to(device=self.device, dtype=getattr(torch, self.settings["dtype"]))
                         if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}
                kwargs = {"return_dict": True}
                if self.family in {"colpali", "colqwen2"}:
                    kwargs["use_cache"] = False
                output = self.model(**moved, **kwargs)
                native = self._numpy(output.embeddings if self.family in {"colpali", "colqwen2"} else output.last_hidden_state)
                require(native.ndim == 3 and native.shape[0] == len(batch_images), "Native output batch shape mismatch")
                result.extend(self._collect(native, inputs, alignment, ids[start:start + size]))
        return result

    @staticmethod
    def _validate_ocr(record):
        words, boxes, confidence = record["words"], record["boxes"], record["word_confidences"]
        require(record["status"] in {"ok", "empty"}, "Failed OCR must not enter inference")
        require(isinstance(words, list) and all(isinstance(w, str) and w.strip() for w in words), "Invalid OCR words")
        require(len(words) == len(boxes) == len(confidence), "OCR word/box/confidence alignment mismatch")
        require((record["status"] == "empty") == (len(words) == 0), "OCR status disagrees with word count")
        require(all(type(v) in {int, float} and np.isfinite(v) and 0 <= v <= 100 for v in confidence), "Invalid OCR confidence")
        require(re.fullmatch(r"[0-9a-f]{64}", record.get("image_sha256", "")) is not None, "OCR image checksum required")
        for box in boxes:
            require(len(box) == 4 and all(type(v) is int and 0 <= v <= 1000 for v in box)
                    and box[0] <= box[2] and box[1] <= box[3], "OCR boxes must be ordered integer coordinates in 0..1000")

    def _prepare(self, images, records):
        if self.mode == "image_only":
            return self.processor(images=images, return_tensors="pt"), None
        # Tokenize words separately so exact untruncated lengths and word IDs are known.
        words, boxes = [r["words"] for r in records], [r["boxes"] for r in records]
        full = self.processor.tokenizer(words, boxes=boxes, truncation=False, padding=False)
        tokens = self.processor.tokenizer(words, boxes=boxes, truncation=True, max_length=512,
                                          padding="longest", return_tensors="pt")
        align = [{"word_ids": tokens.word_ids(batch_index=i), "original_word_count": len(words[i]),
                  "untruncated_text_tokens": len(full["input_ids"][i]),
                  "truncated": len(full["input_ids"][i]) > 512, "ocr_status": records[i]["status"]}
                 for i in range(len(images))]
        pixels = self.processor.image_processor(images=images, return_tensors="pt")
        return {**tokens, **pixels}, align

    def _collect(self, native, inputs, alignment, ids):
        if self.family in {"colpali", "colqwen2"}:
            mask = self._numpy(inputs["attention_mask"])
            token_ids = self._numpy(inputs["input_ids"])
            validate_output_alignment(native, mask, token_ids)
            config = self.model.config.vlm_config
            image_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
            require(image_id is not None, "Checkpoint does not identify its image token")
            special_ids = self.processor.tokenizer.all_special_ids
        else:
            patch = self.model.config.patch_size
            ph, pw = (patch, patch) if isinstance(patch, int) else patch
            height, width = inputs["pixel_values"].shape[-2:]
            patch_count = (height // ph) * (width // pw)
        documents = []
        for i, doc_id in enumerate(ids):
            meta = {"parent_document_id": doc_id, "input_mode": self.mode, "family": self.family}
            if self.family == "dinov3":
                registers = self.model.config.num_register_tokens
                require(native.shape[1] == 1 + registers + patch_count, "DINOv3 CLS/register/patch count mismatch")
                tensors = {"cls": native[i, :1], "registers": native[i, 1:1 + registers],
                           "patches": native[i, 1 + registers:]}
                meta.update(cls_index=0, register_count=registers, patch_grid=[height // ph, width // pw],
                            token_order="cls,registers,row_major_patches")
            elif self.family in {"colpali", "colqwen2"}:
                valid = mask[i].astype(bool)
                image_mask = (token_ids[i] == image_id) & valid
                require(image_mask.any(), "Retrieval image inputs contain no image tokens")
                tensors = {"vectors": native[i], "attention_mask": valid, "input_ids": token_ids[i],
                           "image_mask": image_mask,
                           "special_mask": np.isin(token_ids[i], special_ids) & valid}
                meta.update(prompt_and_special_tokens="retained", padding="retained_with_false_attention_mask",
                            image_token_id=int(image_id), valid_tokens=int(valid.sum()),
                            output_alignment="native_forward_one_vector_per_expanded_input_token")
                if "image_grid_thw" in inputs:
                    meta["image_grid_thw"] = inputs["image_grid_thw"][i].tolist()
            else:
                text_count = inputs["input_ids"].shape[1] if self.mode == "image_and_ocr" else 0
                require(native.shape[1] == text_count + 1 + patch_count, "LayoutLMv3 text/visual output count mismatch")
                output_mask = np.ones(native.shape[1], dtype=bool)
                tensors = {"cls": native[i, :1], "tokens": native[i], "attention_mask": output_mask}
                meta.update(cls_index=0, cls_kind="text_cls" if text_count else "visual_cls",
                            visual_cls_index=text_count, text_token_count=text_count,
                            visual_patch_count=patch_count, token_order="text,visual_cls,row_major_patches")
                if text_count:
                    output_mask[:text_count] = self._numpy(inputs["attention_mask"])[i].astype(bool)
                    tensors.update(input_ids=self._numpy(inputs["input_ids"])[i],
                                   boxes=self._numpy(inputs["bbox"])[i],
                                   word_ids=np.array([-1 if w is None else w for w in alignment[i]["word_ids"]], dtype=np.int64))
                    meta.update(alignment[i])
                    meta["retained_word_count"] = len({w for w in alignment[i]["word_ids"] if w is not None})
            documents.append({"document_id": doc_id, "tensors": tensors, "token_metadata": meta})
        return documents
