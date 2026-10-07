"""Extract native pretrained outputs; defaults to all 1,000 prepared documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dinov3", "colpali", "colqwen2", "layoutlmv3-image", "layoutlmv3-ocr"], default="dinov3")
    parser.add_argument("--config", type=Path, help="Explicit model JSON instead of --model preset")
    parser.add_argument("--storage-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, help="Prepared metadata directory; otherwise requires exactly one dataset")
    parser.add_argument("--document-ids", type=int, nargs="+", help="Explicit pilot IDs; default extracts every manifest ID")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, help="Override configured batch size; changes artifact identity")
    parser.add_argument("--dtype", choices=["float32", "float16"], help="Override precision; changes artifact identity")
    parser.add_argument("--ocr-config", type=Path, help="Actual Tesseract settings for multimodal LayoutLMv3")
    args = parser.parse_args()
    from src.embeddings.pipeline import extract_embeddings

    try:
        settings = json.loads((args.config or EXPERIMENT_ROOT / "configs/models" / f"{args.model}.json").read_text(encoding="utf-8"))
        for key in ("batch_size", "dtype"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            candidates = list((args.storage_root / "outputs/metadata").glob("dataset-*/dataset.json"))
            if len(candidates) != 1:
                raise ValueError("Prepare the dataset first, or select one unambiguously with --dataset-dir")
            dataset_dir = candidates[0].parent
        ocr_config = json.loads(args.ocr_config.read_text(encoding="utf-8")) if args.ocr_config else None
        result = extract_embeddings(settings, args.storage_root, dataset_dir, source_root=EXPERIMENT_ROOT,
                                    document_ids=args.document_ids, device=args.device, ocr_config=ocr_config)
    except Exception as error:
        parser.exit(1, f"Embedding extraction failed ({type(error).__name__}): {error}\n")
    print(json.dumps({key: value for key, value in result.items() if key != "config"}, indent=2))


if __name__ == "__main__":
    main()
