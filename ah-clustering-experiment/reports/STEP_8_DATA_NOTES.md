# Step 8 dashboard data boundary

`dashboard/data.py` reads completed saved metrics and upstream cluster assignments through the existing artifact reader. It never imports clustering, evaluation, embedding runners, encoder libraries or scikit-learn. It reads neither embedding caches nor similarity arrays. NumPy is used only to validate small saved tables and scalar invariants; there is no fitting or projection calculation.

`scan_catalog(root)` returns `runs` and `issues`. The configured evaluation matrix scopes discovery so old pilot artifacts do not pollute the experiment comparison. Configured runs without metrics, incomplete/corrupt metrics, relevant failed experiment attempts, pending OCR and unsupported retained-vector K-means combinations remain explicit issues. Successful publications supersede older failure attempts. Each descriptor includes model, algorithm, representation, cohort, seed, counts, noise percentage, and recorded clustering/evaluation seconds.

`load_run(root, artifact_id)` validates every metrics and assignments payload checksum, immutable artifact identity, upstream manifest reference, cohort membership, integer labels, noise coverage, cluster sizes, label distributions, purity, examples and rank membership, score scopes, finite metric values or unavailable reasons. It aligns both input tables by document ID in the configured cohort order. It does not assume artifact row positions correspond.

`load_projection(root, ref, expected_ids)` validates the saved projection artifact, manifest and method, exact ID membership, coordinates and configured order, then aligns rows for the caller. Projection files are intentionally loaded separately: losing a projection does not hide valid metrics. The UI displays the resulting unavailable reason.

`image_bytes(root, row)` restricts paths to the experiment, checks the recorded image hash when provided, fully decodes the captured bytes with PIL, and returns a PNG. Missing, corrupt, changed or escaping paths fail explicitly. Images are read only on selection; no 1,000-image scan is required to view metrics.

Verification command:

```text
ah-clustering-experiment/.venv/Scripts/python.exe -m unittest discover -s ah-clustering-experiment/tests -p test_dashboard_data.py -v
```

The 13 fixture tests publish actual checksummed artifacts and exercise shuffled rows, wrong labels, duplicate and wrong IDs, incorrect noise coverage, wrong example membership, corruption, incomplete publication, stale failed logs, malformed settings, missing projections, coordinate alignment/fingerprints, and image decode/hash/missing/path escape. These are development self-checks, not evidence that the live interface is usable. Independent verification and the lead's real Streamlit interactions provide that evidence.

A direct real-data smoke inspection loaded all 14 saved metrics and upstream assignments, reporting exactly the three expected pending/unavailable issues. It loaded a 700-row projection and decoded an original image. Source authority, metrics quality and projection fitting remain Step 7's responsibility; this boundary verifies their saved contract and integrity rather than recomputing them.
