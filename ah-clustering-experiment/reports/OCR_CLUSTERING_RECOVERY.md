# LayoutLMv3 OCR clustering recovery

Completed October 6, 2026 (America/Chicago). The dashboard now contains 18 evaluated runs across five input modes, with eight saved projections. The two unavailable entries are the intentional direct K-means exclusions for ColPali and ColQwen2. No OCR or model inference was rerun.

## Changes

- Registered the recovered full cache `embeddings-14f0c9c4cc8cb38ae601` in clustering configuration.
- Ran four OCR clustering algorithms on exactly the shared 700 training documents, using the existing L2-normalized 768-dimensional CLS representation. Every algorithm reproduced its partition on a second fit.
- Registered and evaluated those four artifacts; generated one shared PCA and one shared UMAP projection without refitting clusters during evaluation.
- Replaced the CLI's hardcoded missing-cache claim with a configuration-omission status. It no longer infers cache availability from an absent model mapping.
- Made validated completion of every configured run for a model supersede that model's stale pending note. Missing or corrupt evaluations still retain pending/error status.
- Refreshed the portable dashboard snapshot and current documentation. The original 14 results and all 28 original cluster/metric manifests remain unchanged. Historical reports are preserved as records of their earlier state.

## New results

All scores below include noise. Fixed-count algorithms retain the existing class-count-informed choice of 19 clusters.

| Algorithm | Clusters | Noise documents | ARI | NMI |
| --- | ---: | ---: | ---: | ---: |
| K-means | 19 | 0 | 0.209602 | 0.434779 |
| Agglomerative | 19 | 0 | 0.003266 | 0.148382 |
| HDBSCAN | 11 | 443 | 0.058386 | 0.415891 |
| Spectral | 19 | 0 | 0.201418 | 0.415801 |

HDBSCAN excludes 63.29% of documents as noise; its scores excluding noise cover only 257 documents (36.71%). These are completed experiments, not evidence that OCR improves clustering quality. No parameters were tuned in response to the scores.

| Algorithm | Clustering artifact | Evaluation artifact |
| --- | --- | --- |
| K-means | `clusters-15e00b3a10b0d6815eae` | `metrics-392a579f70f68c84a68f` |
| Agglomerative | `clusters-a63804aa4701271d5c73` | `metrics-b1da1d23b3efcb4940a3` |
| HDBSCAN | `clusters-119f6ff1c9ac8954d1d1` | `metrics-e216df2f5cd09fecdfe4` |
| Spectral | `clusters-3d69dfa5f283ddf9a2d8` | `metrics-13b15ca895dd5d4b5514` |

PCA: `projections-f86380eb4e33f99439a5`. UMAP: `projections-076af2b8068d4957f08a`. Both contain finite coordinates for the same 700 IDs and are shared by the four new evaluations.

## Verification

- Full test discovery: 256 tests, no failures, one skip because the optional live Tesseract executable is unavailable locally. The existing saved OCR cache was consumed without invoking Tesseract.
- A separate verifier froze the original IDs and manifest hashes before implementation. Final checks preserved every original artifact and verified the new assignments and labels against the shared CSV.
- ARI and NMI were independently calculated from contingency counts and mathematical definitions, without the evaluator or scikit-learn agreement functions. Every OCR score in both noise scopes matched within `1e-12`.
- Negative controls detected a missing assignment ID, changed cluster assignment, and changed truth label. Regression fixtures also kept pending warnings for partial or corrupt evaluations.
- Seven encoder import guard checks passed. Compute logs show no source changes during either run and no clustering refit during evaluation.
- The 1,673,801-byte dashboard snapshot contains 212 files, 18 runs, and eight projections. Restoration into isolated storage passed artifact validation; a second restore wrote zero files.
- Streamlit AppTest passed the overview, OCR filter, PCA, UMAP, cluster explorer, and details views. It displayed 18 completed runs and exactly two unavailable entries. No inference or fitting modules loaded. Browser rendering was not separately retested because the change concerns saved data and status logic.

Commands used, from `ah-clustering-experiment`, with the existing local environment:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -B experiments/run_clustering.py --models layoutlmv3-ocr
.venv/Scripts/python.exe -B experiments/run_evaluation.py --models layoutlmv3-ocr
.venv/Scripts/python.exe -B experiments/dashboard_bundle.py --export
.venv/Scripts/python.exe -B reports/ocr-recovery-evidence/audit_dashboard.py
```

The clustering IDs were registered in evaluation configuration between the two compute commands. Source identities include configuration, so those steps intentionally have different digests. After computation, the evaluation JSON's original CRLF format was restored and the portable snapshot index was updated. Those changes explain the final source digest; no inference or clustering implementation changed after the successful runs.

## Evidence ledger

- REVISION: Git `2ea869ec35720d4c244a1a353caf1f4aaf3a032e` plus uncommitted changes. Final runnable source digest: `a549fc50e9d405903300ea6cc05debaafbe9c723afc656e5a25fb62d9f8eec96`.
- CLAIM: The recovered OCR cache has four complete, validated downstream runs, and the dashboard reports their actual completion while preserving the original results and representation restrictions.
- ORACLE_SOURCE: Original shared CSV, prepared cohort/image identities, frozen original artifact hashes, existing experiment requirements, and independently computed agreement metrics.
- REAL_COMPONENTS: Saved embeddings, clustering/evaluation runners, artifact readers, dashboard catalog, snapshot exporter/restorer, and Streamlit Python UI execution.
- ALLOWED_MOCKS: None for the real-run acceptance checks. The broader unit suite includes its existing fixture/mock tests and is supplementary evidence.
- NEGATIVE_CONTROL: Missing IDs, changed assignment/truth labels, partial/corrupt evaluations, and forbidden encoder imports.
- REQUIRED_EVIDENCE_GRADE: INDEPENDENTLY_CHECKED.
- EVIDENCE_GRADE: INDEPENDENTLY_CHECKED.
- CONTRADICTORY_EVIDENCE: None after recovery. The earlier missing-cache message was stale and has been removed from current configuration.
- RESIDUAL_RISKS: No browser visual rendering check, live OCR rerun, Drive durability verification, or claim of improved clustering quality. ColPali/ColQwen2 pooled-vector K-means would be a separate experiment and was not added.
- VERDICT: PASS. Changes remain uncommitted according to the user's Git preference.

Evidence files: [run results and test summary](ocr-recovery-evidence/run-evidence.json), [final source identity](ocr-recovery-evidence/source-identity.json), [dashboard audit](ocr-recovery-evidence/dashboard-audit.json), and [independent audit](ocr-recovery-evidence/independent-audit.json).
