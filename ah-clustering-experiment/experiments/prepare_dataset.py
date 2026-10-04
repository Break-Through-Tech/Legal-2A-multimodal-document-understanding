"""Prepare original images and the shared clustering cohort from either cwd."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
# Do not create bytecode caches beside experiment source or outside its root.
sys.dont_write_bytecode = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs/dataset.json")
    parser.add_argument("--storage-root", type=Path, default=EXPERIMENT_ROOT, help="Persistent directory named ah-clustering-experiment")
    parser.add_argument("--split-csv", type=Path, help="Shared assignment CSV; defaults to configured repository-relative path")
    parser.add_argument("--cohort", choices=["train", "val", "test", "all"], help="Default: configured training cohort; all is explicitly exploratory")
    args = parser.parse_args()
    from src.dataset import prepare_dataset

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assignments_path = args.split_csv or EXPERIMENT_ROOT.parent / config["shared_manifest"]
    try:
        result = prepare_dataset(config, args.storage_root, assignments_path, cohort=args.cohort)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
