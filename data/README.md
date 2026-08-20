# Dataset

**OmniAI OCR Benchmark** — 1,000 business document images with ground-truth structured JSON.
MIT licensed, ~360 MB. https://huggingface.co/datasets/getomni-ai/ocr-benchmark


## Setup

```bash
pip install datasets pandas pillow
```

```bash
python data/load_data.py --summary
```

This writes `labels.csv` (one row per document, with `format` and `document_quality` parsed out of `metadata`) and caches the dataset in `raw/`. The `--summary` flag prints the class distribution.

## Loading it yourself

```python
from datasets import load_dataset

ds = load_dataset("getomni-ai/ocr-benchmark", split="test")
```

The dataset ships as a single `test` split of 1,000 rows. Build your own train/val/test split, stratified by `format`, with a fixed seed.

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
3. **Severe class imbalance.** 18 classes hold 925 rows; 18 more share 75, several with a single example. Decide explicitly what to do with the rare ones.
4. **This is a general business-document benchmark, not a legal one.** Roughly 417 of the 925 well-populated documents are legal or legal-adjacent. Which types you treat as in scope is a decision you make and defend — see the Legal Subset section in [`../Challenge-Project-Overview.md`](../Challenge-Project-Overview.md). The loader deliberately does not make this call for you; it just parses labels.
5. **Images vary enormously in size.** Cap the long edge on resize before batching.
