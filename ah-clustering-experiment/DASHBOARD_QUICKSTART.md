# Clone the repository and run the dashboard

The repository includes a compressed snapshot of 18 evaluated clustering runs and eight projections for 700 training documents. You can inspect these results on a CPU without training models, downloading model weights or using Colab. The snapshot includes all five input modes, including LayoutLMv3 with OCR.

## Get the code

Clone the repository directly. Until this work is merged into the default branch, select `anh-embedding-experiments`:

```text
git clone --branch anh-embedding-experiments https://github.com/Break-Through-Tech/Legal-2A-multimodal-document-understanding.git legal2a
cd legal2a
```

Use Python 3.12 or newer; the clean installation was tested with Python 3.14. The pinned NumPy release does not support Python 3.11.

## Windows PowerShell

Run these commands from the repository root. Activation is unnecessary. Use a short clone location such as `C:/src/` if Windows reports a path-length error while creating the environment or installing packages. Check `python --version` first; it must be at least 3.12.

```powershell
python -m venv ah-clustering-experiment/.venv
ah-clustering-experiment/.venv/Scripts/python.exe -m pip install -r ah-clustering-experiment/requirements-dashboard.txt
ah-clustering-experiment/.venv/Scripts/python.exe ah-clustering-experiment/experiments/dashboard_bundle.py
ah-clustering-experiment/.venv/Scripts/python.exe ah-clustering-experiment/experiments/prepare_dataset.py
ah-clustering-experiment/.venv/Scripts/python.exe -m streamlit run ah-clustering-experiment/dashboard/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

## macOS or Linux

```bash
python3 -m venv ah-clustering-experiment/.venv
ah-clustering-experiment/.venv/bin/python -m pip install -r ah-clustering-experiment/requirements-dashboard.txt
ah-clustering-experiment/.venv/bin/python ah-clustering-experiment/experiments/dashboard_bundle.py
ah-clustering-experiment/.venv/bin/python ah-clustering-experiment/experiments/prepare_dataset.py
ah-clustering-experiment/.venv/bin/python -m streamlit run ah-clustering-experiment/dashboard/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://localhost:8501. Stop Streamlit with Ctrl+C. If port 8501 is in use, append `--server.port 8502` and open that port instead.

## Results and images

`dashboard_bundle.py` restores the committed `dashboard/saved-results.zip` into the ignored `outputs/` directory. It checks the archive and every file, preserves artifact identities, and refuses to overwrite different local results. Running it again safely reuses identical files.

`prepare_dataset.py` downloads the original images from `getomni-ai/ocr-benchmark` at the pinned revision in `configs/dataset.json`. It uses the repository's exact `data/split.csv` and verifies image contents. This requires internet access and disk space for the dataset and download cache, but no GPU. Dataset access remains subject to the upstream dataset's terms.

You can skip `prepare_dataset.py` to inspect all scores, projections, cluster tables and comparisons offline after installing dependencies. Image previews will show a missing-image message until the dataset is prepared. Do not edit or resave `data/split.csv`: its exact bytes are part of the saved dataset identity.

The snapshot contains result tables and provenance, not model weights, embeddings, similarity matrices, source document images or reference transcriptions. It supports viewing the saved experiments. Recomputing experiments requires the extraction and evaluation setup in [README.md](README.md).

## Updating the snapshot

After publishing and verifying new local results, maintainers can run:

```text
python ah-clustering-experiment/experiments/dashboard_bundle.py --export
```

Commit both `dashboard/saved-results.zip` and `dashboard/saved-results.json` with the code and configuration changes. Keep generated caches and the expanded `outputs/` directory out of Git.
