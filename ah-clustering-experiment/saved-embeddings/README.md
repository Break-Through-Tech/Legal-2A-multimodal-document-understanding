# Use the computed embeddings

This branch includes all five completed embedding runs through Git LFS. Each run
covers all 1,000 documents and preserves the original native tensor values,
document IDs, masks, configuration, provenance, and checksums. No encoder inference
or GPU is needed to restore or read them.

Archives are split into parts of at most 8 MiB for parallel downloads and smaller
retries. The restore command assembles the parts and checks their SHA-256 hashes.

| Model | Artifact ID |
| --- | --- |
| DINOv3 | `embeddings-9ef1694d986576ba0acf` |
| LayoutLMv3 image | `embeddings-af5fe5262fbc515a18be` |
| LayoutLMv3 OCR | `embeddings-14f0c9c4cc8cb38ae601` |
| ColPali | `embeddings-d987f31cbdb073fed394` |
| ColQwen2 | `embeddings-17d7e1b5ec380770c824` |

The archives also include the dataset manifest needed to join document IDs to
labels and partitions. The OCR model includes its referenced OCR artifact.
Original document images, model weights, pilot runs, and restart checkpoints are
excluded. The full extraction report is in
[`reports/FULL_EMBEDDING_RUN.md`](../reports/FULL_EMBEDDING_RUN.md).

## Download and restore

Install [Git LFS](https://git-lfs.com/) first, then run:

```text
git lfs install
git clone --branch anh-embedding-experiments https://github.com/Break-Through-Tech/Legal-2A-multimodal-document-understanding.git legal2a
cd legal2a
git lfs pull
python ah-clustering-experiment/experiments/restore_embeddings.py
```

For an existing checkout of this branch, run `git pull --ff-only`, `git lfs pull`,
and the restore command. Restoration uses only the Python 3.12+ standard library.
It verifies each archive and every expanded file against `index.json` and refuses
to overwrite different existing files. Matching files are reused. Use one restore
process per destination at a time.

Allow about 14 GB of free space for the LFS cache, downloaded archive parts, expanded
artifacts, and temporary extraction files. Download and storage usage count toward
the repository owner's [Git LFS allowance](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).

To restore just ColPali after downloading the archives:

```text
python ah-clustering-experiment/experiments/restore_embeddings.py --models colpali
```

`--storage-root /path/to/ah-clustering-experiment` restores to a different location.
The final directory must be named `ah-clustering-experiment`. To download only a
chosen model, clone with the `GIT_LFS_SKIP_SMUDGE=1` environment variable set, then
run `git lfs pull --include="ah-clustering-experiment/saved-embeddings/colpali.zip.part-*"`.

## Read the tensors

Install the storage dependencies with
`python -m pip install -r ah-clustering-experiment/requirements.txt`, then run this
from the repository root:

```python
import json
from pathlib import Path
import sys

root = Path("ah-clustering-experiment").resolve()
sys.path.insert(0, str(root))
from src.embeddings import read_embeddings

artifact_id = "embeddings-d987f31cbdb073fed394"
config = json.loads(
    (root / "outputs" / "metadata" / artifact_id / "config.json").read_text()
)
documents = read_embeddings(root, config)
print(len(documents))  # 1000
print(documents[0]["tensors"]["vectors"].shape)
```

The reader validates the saved artifacts and returns tensors keyed by document ID.
Use `configs/clustering.json` for the other model IDs and
[`ARTIFACT_FORMAT.md`](../ARTIFACT_FORMAT.md) for tensor layouts. Downstream
clustering can read these caches directly. Do not run `extract_embeddings.py` to
load them; its environment and source identity checks can request a new extraction.

The shared partitions remain 700 training, 150 validation, and 150 test documents.
Loading embeddings for every partition does not authorize fitting on held-out
documents. The existing reports disclose prior exploratory use of the test set.
Source images are needed only for image inspection and can be downloaded using
the [dashboard setup](../DASHBOARD_QUICKSTART.md).
