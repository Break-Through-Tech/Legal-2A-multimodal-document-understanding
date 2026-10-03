# Dataset

**OmniAI OCR Benchmark** — 1,000 business document images with ground-truth structured JSON.
MIT licensed, ~360 MB. https://huggingface.co/datasets/getomni-ai/ocr-benchmark


## Setup

Use the shared [`split.csv`](split.csv) directly. **You do not need to run
`data/load_data.py` or generate `labels.csv`.** The CSV supplies the final labels
and train/validation/test assignments; Hugging Face supplies the document images
and reference data.

```bash
pip install datasets pandas pillow
```

## Load the shared dataset

Run from the repository root. In Colab, upload `split.csv` and change the CSV
path below to `/content/split.csv`.

```python
import pandas as pd
from datasets import load_dataset

splits = pd.read_csv("data/split.csv")
ds = load_dataset(
    "getomni-ai/ocr-benchmark",
    revision="4ed0d95271ca00107726230f7a0944ed9e90d897",
    split="test",
)

assert splits.id.is_unique and set(splits.id) == set(ds["id"])
assert splits["split"].isin(["train", "val", "test"]).all()
by_id = splits.set_index("id")
ds = ds.add_column("label", [by_id.at[doc_id, "label"] for doc_id in ds["id"]])
ds = ds.add_column("split", [by_id.at[doc_id, "split"] for doc_id in ds["id"]])

# Select by the shared assignments, without decoding images to filter rows.
partitions = {
    name: ds.select([i for i, assignment in enumerate(ds["split"]) if assignment == name])
    for name in ("train", "val", "test")
}
train, val, test = (partitions[name] for name in ("train", "val", "test"))
```

The dataset ships as a single upstream `test` split of 1,000 rows. That name does
not refer to our team's test partition. Use the shared [`split.csv`](split.csv)
for team experiments so everyone uses the same documents, labels, and assignments.
The revision above is the dataset version used to create this file.

The existing `data/load_data.py --summary` command is an optional exploration
utility that writes the original format labels to `labels.csv`. It is not part
of the shared loading workflow.

## Shared split and labels

The source has **36 normalized format classes** after stripping whitespace and
uppercasing labels:

- **18 head classes:** 925 documents, with 49–59 documents per class. Formats
  with at least 30 documents retain their original normalized label.
- **18 tail classes:** 75 documents total. All of these documents are assigned
  the case-sensitive label **`Unknown`**.

This gives **19 final classes**. The number 75 refers to tail documents, not
tail classes. The CSV preserves each document's original normalized label.

### How the 70/15/15 split was made

The proportions follow the project plan. Tail labels were merged into `Unknown`
**before** splitting, then the documents were split as follows:

1. Sort all documents by their original dataset ID for reproducibility.
2. Use scikit-learn's `train_test_split` with `stratify` set to the final
   19-class label, `random_state=42`, `train_size=700`, and `test_size=300`.
3. Split the resulting 300-document holdout with `train_test_split`, again
   stratifying by the final label and using `random_state=42`, with 150
   documents assigned to validation and 150 to test.

The result is **700 train / 150 validation / 150 test** documents. Each ID
appears exactly once, and every partition contains all 19 final classes.
Stratification approximately preserves each class's proportion; small differences
from 70/15/15 within a class are necessary because documents are whole numbers.
No documents were oversampled or discarded. Generation used scikit-learn 1.9.1.

The final CSV is the shared source of assignments. The split-generation code is
not included; teammates should load this file instead of recreating splits.

### Exact class distribution

| Final label | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| ACCOUNT_STATEMENT | 36 | 8 | 8 | 52 |
| BANK_CHECK | 37 | 8 | 7 | 52 |
| CHART | 34 | 7 | 8 | 49 |
| COMMERCIAL_LEASE_AGREEMENT | 36 | 8 | 8 | 52 |
| CREDIT_CARD_STATEMENT | 35 | 8 | 7 | 50 |
| DELIVERY_NOTE | 36 | 7 | 8 | 51 |
| EQUIPMENT_INSPECTION | 35 | 7 | 8 | 50 |
| FORM_1040 | 36 | 8 | 7 | 51 |
| GLOSSARY | 35 | 7 | 8 | 50 |
| NUTRITION | 34 | 7 | 8 | 49 |
| PATENT | 36 | 8 | 8 | 52 |
| PATIENT_INTAKE | 37 | 8 | 8 | 53 |
| PAY_IN_SHEET | 35 | 8 | 7 | 50 |
| PETITION_FORM | 36 | 8 | 7 | 51 |
| PROXY_VOTING | 35 | 8 | 7 | 50 |
| REAL_ESTATE | 41 | 9 | 9 | 59 |
| SHIFT_SCHEDULE | 36 | 8 | 8 | 52 |
| SHIPPING_INVOICE | 37 | 7 | 8 | 52 |
| **Unknown** | **53** | **11** | **11** | **75** |
| **Total** | **700** | **150** | **150** | **1,000** |

### Using the CSV

| Column | Meaning |
|---|---|
| `id` | Original dataset document ID; join on this, not row position |
| `split` | `train`, `val`, or `test` |
| `label` | Final target class: a head format or `Unknown` |
| `original_label` | Normalized source format, retained for analysis |
| `document_quality` | Normalized document quality metadata |

Use `label` as the classification target. `labels.csv` from the existing loader
contains the original format labels and does not replace this shared manifest.
The CSV contains assignments and labels; images remain in the source dataset.

### Interpretation and limitations

- `Unknown` is a **supervised class** with examples in train, validation, and
  test. This evaluates recognition of the merged class, not rejection of wholly
  unseen document types.
- The 18 original tail formats are **not individually stratified**. A rare
  format may appear in only one partition, including a format with one document.
- Splitting is at the document level. The existing image audit found no exact
  RGB-pixel duplicates, but related source/template families were not grouped;
  similar layouts may appear across partitions.
- The dataset has already been explored and used in pilots. This new split
  does not preserve earlier five-class pilot assignments and is not an untouched
  holdout. Existing notebooks using `splits.csv` (plural) or filtering to five
  classes need to explicitly adopt `data/split.csv` and its final `label` column
  for the shared 19-class experiment.

## Columns

| Column | Type | Notes |
|---|---|---|
| `id` | int64 | 0–999 |
| `image` | image | PIL image, widths 160–6,910 px |
| `metadata` | string | JSON string with `format` and `documentQuality`. Your labels are in here. |
| `json_schema` | string | The JSON schema for this specific document |
| `true_json_output` | string | Ground-truth structured extraction |
| `true_markdown_output` | string | Ground-truth transcription |

## Known issues

1. **Labels are nested in a JSON string.** Parse `metadata` first. The loader does this.
2. **`"SCANNED_TABLE "` has a trailing space** and reads as a separate class from `"SCANNED_TABLE"`. The loader strips whitespace, taking the label count from 37 to 36. Assume there are other issues nobody has caught yet.
3. **Severe imbalance in the original labels.** 18 classes hold 925 rows; 18 more share 75, several with a single example. The shared split merges those 75 documents into `Unknown`, as described above.
4. **This is a general business-document benchmark, not a legal one.** Roughly 417 of the 925 well-populated documents are legal or legal-adjacent. Which types you treat as in scope is a decision you make and defend — see the Legal Subset section in [`../Challenge-Project-Overview.md`](../Challenge-Project-Overview.md). The loader deliberately does not make this call for you; it just parses labels.
5. **Images vary enormously in size.** Cap the long edge on resize before batching.
