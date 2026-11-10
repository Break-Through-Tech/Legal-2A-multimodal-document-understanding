# AI Studio Challenge Project Title

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Taylor Nguyen    | @taylornguyen | Data exploration, visualization, overall project coordination            |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.

---

# Document-intake pipeline (v1)

Scanned image → predicted type (18 types + `Unknown` = out of scope) → human-review flag → extracted, schema-validated JSON.
Model: SigLIP `google/siglip-base-patch16-224` embeddings + logistic regression, with test-time rotation (0/90/180/270, most confident angle wins).
Data: HuggingFace `getomni-ai/ocr-benchmark` (1000 images, downloaded on first use).

## Run (WSL, `.venv` active, `pip install -r requirements.txt`)
Put `split.csv ids.npy labels.csv siglip_siglip-base-patch16-224.npy siglip_rot4.npy` in `data/embeddings/` (gitignored).

| Step | Command |
|---|---|
| 1. Train classifier + flag table | `python classify_and_flag.py data/embeddings` → `outputs/classifier.joblib`, `outputs/predictions.csv` |
| 2. Embed an image (self-check vs cache) | `python embed.py` |
| 3. Predict (self-check vs predictions.csv) | `python pipeline.py` |
| 4a. Build per-class schemas | `python build_schemas.py data/embeddings` → `schemas/<CLASS>.json` |
| 4b. Try extraction (live LLM) | `python test_extraction.py data/embeddings --per-class 3` |
| 5. Split → `results.csv` | `python run_all.py [--split test] [--limit N] [--no-extract]`; saved after every document, re-runs skip finished ids (Groq free tier is rate-limited: use `--limit`) |
| 5b. Offline test (fake LLM) | `python test_run_all.py` |
| 6. Demo UI | `streamlit run app.py` |

Extraction uses a vision LLM through an OpenAI-compatible API (Groq by default, `extract_grok.py`). Put `LLM_API_KEY` and `LLM_MODEL` in `.env` (gitignored; never commit).

## Results (test split, 150 docs)
| Metric | Value | Note |
|---|---|---|
| Macro-F1 (with rotation) | 0.993 | no rotation: 0.980 |
| Sent to human review | 9.3% (14/150) | all 11 `Unknown` docs caught; 3 in-scope low-confidence |
| C = 100 | – | **TODO tune** |
| Review threshold = 0.70 | – | **TODO tune** |
| JSON validity rate | not run yet | run `run_all.py` with the real LLM |
| Mean field accuracy | not run yet | same |

## Notes
- Dataset schemas are per document; `build_schemas.py` merges them into one schema per class (`schemas/`), used by `run_all.py` and the app. `Unknown` has no schema and goes straight to human review.
- Review reasons: low confidence (< 0.70), predicted Unknown, extraction empty (filled ratio < 0.30).
- Deferred work: see `TODO.md`.
