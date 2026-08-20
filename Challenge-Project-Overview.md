# Legal Document Intake: Comparing Multimodal Embeddings for Classification and Extraction

**Company / Org:** Independent project, legal domain focus  
**Challenge Advisor:** Grace Lang, graceelang@gmail.com  
**AI Studio Coach:** Parth Dali, parth.dali@breakthroughtech.org  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About This Project

Law firms and in-house teams process large volumes of scanned documents during due diligence, litigation, and lease review, and the documents look alike: leases, deeds, patents, and court filings are all dense text with headers and signature blocks. Telling them apart is a difficult problem.

---

## 🎯 The Challenge

### Project Summary
Build a pipeline that takes a scanned document arriving in a legal review queue and produces structured data from it: embed → cluster/classify → flag unrecognized documents → extract JSON. Along the way, answer an open research question: do multimodal vision embeddings separate legal document types better than embeddings built from OCR text?

### Why This Matters
Legal teams pay people to look at each incoming document, decide what it is, route it, and retype the important fields. This is called intelligent document processing (IDP). Vision models that read document images directly, without a separate OCR step, are new enough that whether they beat OCR is still open.

For legal documents specifically, the answer is not obvious. Vision embeddings do well when document types look different from each other. Legal documents mostly do not. If layout stops carrying the signal, text may matter more. That is the question you are testing.

### December Deliverables
- A reproducible end-to-end pipeline from document image to structured JSON
- Evaluation results for each stage, with a documented baseline for each
- A quantitative answer to the embedding model vs. OCR-text question
- A portfolio-ready repository with documentation, results, and example outputs
- Final presentation

### Success Criteria
By the end of the semester, the team should be able to demonstrate:
- Document type labels parsed out of the raw dataset and a documented policy for which types count as in-scope
- A head-to-head embedding comparison with metrics and visualizations
- Clustering evaluated against true labels across at least three algorithms
- A classifier over in-scope legal document types, with a documented baseline and at least one measured improvement
- A detector that flags documents falling outside the legal types the classifier was trained on
- Structured JSON extraction scored against ground truth, with error analysis
- Documented limitations

### Technical Scope
1. **Ingestion and labeling** — load the dataset, parse labels, decide scope, build train/val/test splits
2. **Embedding** — encode documents with open source embedding models (ex. ColPali, ColQwen2) and an OCR-text channel; cache the results
3. **Clustering** — compare algorithms and measure agreement with true labels
4. **Classification** — predict document type from embeddings
5. **Out-of-scope detection** — flag documents that are not one of the legal types
6. **Extraction** — prompt a multimodal LLM for structured JSON using each document's schema
7. **Validation** — check extracted JSON against its schema deterministically

### Recommended Modeling Progression
Start simple and add complexity only when evaluation shows a clear benefit:
1. Majority-class baseline for classification, random baseline for clustering
2. OCR text plus TF-IDF and a linear model
3. Pooled multimodal document embeddings plus a simple classifier or clustering method
4. Tuned classifiers and pooling variants
5. Multimodal LLM extraction, once classification and evaluation are working

### Stretch Goal
An agentic intake router: a tool-calling agent that classifies an incoming document, flags out-of-scope documents for human review, routes recognized types to the right extraction schema, and validates the resulting JSON. Attempt this only after the core pipeline is measured and working.

### Project Milestones

Use these milestones to guide your work. Your team should maintain a GitHub Projects board to break these monthly goals into weekly tasks.

| Month | Focus | Expected Outcomes |
|---|---|---|
| **September** | Data understanding and embeddings | Labels parsed, scope policy documented, splits built, all embedding channels cached, clustering compared, head-to-head verdict delivered |
| **October** | Classification and out-of-scope documents | Classifier trained and tuned on legal types, confusion-matrix analysis, out-of-scope detector built and evaluated, first error analysis |
| **November** | Extraction and evaluation | Multimodal LLM extraction working, deterministic schema validation, field-level scoring against ground truth, error analysis by document quality |
| **December** | Integration and presentation | End-to-end pipeline, final results tables, documented limitations, portfolio-ready repository, final demo |

> **Note for the team:** Create a GitHub Projects board in this repository and track work by issue. Recommended columns: Backlog, This Week, In Progress, In Review, Done.

---

## 📊 Dataset

**OmniAI OCR Benchmark** — 1,000 business document images with ground-truth structured JSON.

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/getomni-ai/ocr-benchmark |
| **Size** | ~360 MB |
| **License** | MIT |
| **Split** | A single `test` split of 1,000 rows |

See [`data/README.md`](data/README.md) for a loader script and setup notes.

### Columns

| Column | Type | Notes |
|---|---|---|
| `id` | int64 | 0–999 |
| `image` | image | Widths range 160–6,910 px |
| `metadata` | string | JSON string containing `format` and `documentQuality`. Your labels are in here. |
| `json_schema` | string | The JSON schema for this specific document |
| `true_json_output` | string | Ground-truth structured extraction; the extraction target |
| `true_markdown_output` | string | Ground-truth transcription; a clean-OCR upper bound |

### The Legal Subset

This is a general business-document benchmark, but a large share of it is legal or legal-adjacent. The 18 well-populated document types split roughly as follows:

| Group | Types | Documents |
|---|---|---|
| **Core legal** | `REAL_ESTATE` (59), `COMMERCIAL_LEASE_AGREEMENT` (52), `PATENT` (52), `PETITION_FORM` (51), `PROXY_VOTING` (50) | 264 |
| **Financial / discovery** | `ACCOUNT_STATEMENT` (52), `CREDIT_CARD_STATEMENT` (50), `FORM_1040` (51) | 153 |
| **Everything else** | `PATIENT_INTAKE` (53), `BANK_CHECK` (52), `SHIPPING_INVOICE` (52), `SHIFT_SCHEDULE` (52), `DELIVERY_NOTE` (51), `EQUIPMENT_INSPECTION` (50), `GLOSSARY` (50), `PAY_IN_SHEET` (50), `CHART` (49), `NUTRITION` (49) | 508 |

The suggested framing is to treat the legal types as your in-scope classification problem and the remaining types as documents that should be flagged rather than classified. That mirrors the real workflow: a document arrives in a review queue, and the system either recognizes it or routes it to a person.


### Working Dataset Expectations
- The full dataset fits comfortably on Colab free tier. You should not need to subsample, but if you do, document it.
- Cache embeddings to disk or Google Drive. Roughly 260 MB per model in fp16. Recomputing them is the largest avoidable time cost in this project.

### Known Preprocessing and Data Risks
- **The class distribution is severely long-tailed.** There are 36 class types. Eighteen hold 925 of the 1,000 rows; the other eighteen share 75, several with a single example. You cannot train or evaluate on a class with one example, so decide explicitly what to do with them.
- **There is no train/val/test split.** Build your own, stratify by class type, and fix your seed.

### Data Documentation the Team Must Produce
- Your in-scope definition and the reasoning behind it
- Counts for documents and labels retained after filtering
- The train/val/test split strategy and seed
- Pinned versions for `transformers` and open source embedding models

---

## 🛠️ Suggested Approach

**ML Problem Type:** Computer vision / multimodal embeddings / clustering / classification / out-of-distribution detection / structured extraction

**Recommended Libraries and Tools**
- Python, pandas, NumPy
- scikit-learn — clustering, classifiers, novelty detection, metrics
- Hugging Face Transformers, plus whatever loader your chosen embedding model needs
- `umap-learn` — embedding visualization
- `pytesseract` or `easyocr` — the OCR text channel
- `jsonschema` — deterministic validation of extracted JSON
- Google Colab for experiments; Google Drive for the embedding cache

### Choosing Your Models

This project uses open source models, meaning ones you download and run yourself: they are free, your results stay reproducible, and documents never leave your machine, which matters in legal work — whereas hosted API models like Claude, GPT, or Gemini are often stronger but cost money per call and send your data to a vendor.

Picking specific models is part of the work, not something to inherit from this document. Two families are worth starting from:

- **ColPali family** — vision document embeddings built on PaliGemma
- **Qwen-VL family** — including ColQwen variants for embeddings, and Qwen-VL instruct models for the extraction stage

Both families release new versions regularly, and better options may exist by September. Check current rankings and pick deliberately:

- Visual document retrieval: https://huggingface.co/spaces/vidore/vidore-leaderboard
- Text embeddings, for your OCR channel: https://huggingface.co/spaces/mteb/leaderboard
- Browse by task: https://huggingface.co/models?pipeline_tag=visual-document-retrieval

Whatever you choose, record the exact model IDs and versions in your README, and explain why you chose them.

### Compute Notes
- Colab free tier gives you a T4 with 16 GB of VRAM. Everything here fits if you are deliberate.
- Colab disconnects. Write your embedding cache incrementally, not at the end of the loop.
- For extraction, start with a roughly 3B-parameter vision model. Larger models need quantization to fit. Check what the current small-model options are in September; this area moves fast.

### Evaluation Metrics

Determining evaluation metrics is up to you, but here are some suggestions:

| Component | Primary Metrics | What the Metric Checks |
|---|---|---|
| Clustering | Silhouette, Davies–Bouldin, ARI, NMI | Whether clusters are separated and match true types |
| Classification | Macro F1, per-class F1, confusion matrix | Whether confusable legal types are actually distinguished |
| Out-of-scope detection | Precision, recall, ROC-AUC | Whether non-legal documents are actually flagged |
| Extraction | Field-level accuracy and F1, schema validity rate | Whether output is correct and well-formed |

Validate extracted JSON with `jsonschema`, not by asking the LLM whether its own output is valid.

---

## 📚 Resources to Get Started

The resources below are enough to start productively without overloading the first month.

**Background Reading**
- What intelligent document processing is: https://aws.amazon.com/what-is/intelligent-document-processing/
- Benchmark methodology from the dataset authors, including their JSON scoring formula: https://github.com/getomni-ai/benchmark/blob/main/README.md

**Legal Domain Context**
- Building image understanding for legal documents: https://www.harvey.ai/blog/building-image-understanding-for-legal-documents
- Scaling document processing in a legal platform: https://www.harvey.ai/blog/scaling-document-processing-across-harvey

**Technical Tutorials**
- scikit-learn clustering guide: https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn novelty and outlier detection: https://scikit-learn.org/stable/modules/outlier_detection.html
- Loading image datasets in Hugging Face: https://huggingface.co/docs/datasets/en/image_load
- How UMAP works, and what its plots cannot tell you: https://umap-learn.readthedocs.io/en/latest/how_umap_works.html

**Choosing Models**
- Visual document retrieval leaderboard: https://huggingface.co/spaces/vidore/vidore-leaderboard
- Text embedding leaderboard, for the OCR channel: https://huggingface.co/spaces/mteb/leaderboard
- Browse visual document retrieval models: https://huggingface.co/models?pipeline_tag=visual-document-retrieval

**Code Examples**
- `colpali-engine`, with runnable inference examples: https://github.com/illuin-tech/colpali
- Example model cards, useful for seeing the usage pattern rather than as recommendations: https://huggingface.co/vidore/colqwen2-v1.0 and https://huggingface.co/vidore/colpali-v1.3
- `jsonschema` documentation: https://python-jsonschema.readthedocs.io/en/stable/

**Project Management**
- GitHub Projects documentation: https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Your team's channel in the Break Through Tech Discord, or email (please copy your teammates and your AI Studio Coach)  
**Response time:** Within 48 hours on weekdays. For anything urgent, go to your AI Studio Coach first.

**Recommended Tools**
- **Coding:** Google Colab, VS Code, Jupyter notebooks
- **Collaboration:** GitHub Issues, GitHub Projects
- **Virtual Meetings:** Zoom, Google Meet

### What I Expect From the Team
- Keep work visible in GitHub Issues and the Projects board
- Record modeling decisions and failed experiments, not only successful ones
- Cache embeddings, set seeds, and pin versions so results are reproducible
- Move reusable code out of notebooks and into Python modules as the project matures
- When you are stuck, send the actual error and what you already tried

---

## 🚀 Getting Started

1. Read this overview and list your open questions before our first meeting.
2. Load the dataset using [`data/README.md`](data/README.md) and look at twenty documents before writing any modeling code. Pay attention to how similar the legal types look to each other.
3. Parse `metadata` into real columns and reproduce the class counts above. If your numbers differ, say so.
4. Decide and document which document types are in scope.
5. Create your GitHub Projects board and open initial issues for data loading, EDA, the evaluation harness, and baselines.

---

## ❓ Questions?

Bring questions to our first meeting during the week of August 24th (Bridge to Studio, Session C). Before then, use Discord or email if you hit a blocker on data access, scope, or setup.
