"""Run in its own process: AppTest exercises Streamlit Python behavior, not browser rendering."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dashboard.data import scan_catalog


def main():
    rows = scan_catalog(ROOT)["runs"]
    lookup = {(r["model"], r["algorithm"]): r["artifact_id"] for r in rows}
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=90).run()
    checks = []

    def clean(name):
        assert not app.exception, [e.message for e in app.exception]
        assert not app.error, [e.value for e in app.error]
        checks.append(name)

    def widget(kind, label):
        return next(x for x in getattr(app, kind) if x.label == label)

    def page(name):
        app.radio(key="page").set_value(name).run()
        clean(name)

    clean("initial overview")
    assert any(x.label == "Completed runs" and x.value == "14" for x in app.metric)
    for noise, scope in (("Include noise", "including_noise"), ("Exclude noise", "excluding_noise")):
        app.radio(key="noise_policy").set_value(noise).run()
        clean("overview " + noise)
        # The expanded full table is independently compared to saved score JSON.
        full = next(x.value for x in app.dataframe if "Artifact" in x.value.columns)
        assert len(full) == 14
        for _, row in full.iterrows():
            raw = json.loads((ROOT / "outputs/metrics" / row.Artifact / "scores.json").read_text())[scope]
            assert row.ARI == raw["metrics"]["ari"]["value"]
            assert row["Coverage %"] == raw["coverage"] * 100
    page("Embedding explorer")
    plot = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert plot["layout"]["clickmode"] == "event+select"
    for trace in plot["data"]:
        assert all(name in trace["hovertemplate"] for name in ("Algorithm", "Checkpoint", "Run"))
        assert all(row[-1] == "clusters-7c58a2ef2e2dc1e1560a" for row in trace["customdata"])
    checks.append("projection hover exposes correct run identity and enables point selection")
    widget("selectbox", "Projection").select("UMAP").run()
    clean("choose UMAP")
    for mode in ("colpali", "layoutlmv3-image", "colqwen2", "dinov3"):
        app.selectbox(key="active_run").set_value(lookup[(mode, "hdbscan")]).run()
        clean("projection run switch " + mode)
        for color in ("Cluster", "Source", "Review status"):
            app.selectbox(key="projection_color").set_value(color).run()
            clean(mode + " color " + color)
        app.checkbox(key="explorer_noise").set_value(False).run()
        clean(mode + " exclude noise")
        widget("selectbox", "Document").select_index(-1).run()
        clean(mode + " last visible image")
    app.multiselect(key="explorer_labels").set_value(["Unknown"]).run()
    clean("Unknown label filter")
    assert any("Unknown" in x.value for x in app.caption)
    page("Cluster explorer")
    widget("selectbox", "Cluster").set_value(-1).run()
    clean("HDBSCAN noise cluster")
    app.radio(key="cluster_subset").set_value("Representatives").run()
    clean("noise representatives empty")
    assert any("No documents" in x.value for x in app.info)
    app.selectbox(key="active_run").set_value(lookup[("dinov3", "kmeans")]).run()
    clean("noise to K-means cluster options")
    for selection in ("All members", "Representatives", "Candidate outliers", "Majority-label mismatches"):
        app.radio(key="cluster_subset").set_value(selection).run()
        clean("cluster subset " + selection)
    page("Experiment comparison")
    widget("selectbox", "Compare against").select_index(-1).run()
    clean("comparison alternative")
    app.multiselect(key="comparison_status").set_value([]).run()
    clean("comparison all statuses")
    assert any(len(x.value) == 700 and "status" in x.value.columns for x in app.dataframe)
    app.selectbox(key="active_run").set_value(lookup[("colpali", "hdbscan")]).run()
    clean("comparison run switch with current options")
    page("Experiment details")
    score = next(x.value for x in app.dataframe if "Unavailable reason" in x.value.columns)
    assert score.loc[score.Metric == "Davies–Bouldin", "Unavailable reason"].iloc[0]
    page("Overview")
    app.multiselect(key="models").set_value([]).run()
    clean("zero model filters")
    assert any("Select at least one" in x.value for x in app.info)
    app.multiselect(key="models").set_value(["colqwen2"]).run()
    clean("restore restricted mode")
    assert any(x.label == "Completed runs" and x.value == "3" for x in app.metric)
    for name in ("torch", "transformers", "sklearn", "src.embeddings.models", "src.clustering.runner", "src.evaluation.runner"):
        assert name not in sys.modules, name
        try:
            importlib.import_module(name)
        except ImportError as error:
            assert "saved results only" in str(error), str(error)
        else:
            raise AssertionError("Dashboard import guard allowed " + name)
    checks.append("real inference/fitting import attempts blocked")
    evidence = {"status": "pass", "surface": "Streamlit AppTest Python execution; browser rendering is separate",
                "check_count": len(checks), "checks": checks,
                "dashboard_file_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted((ROOT / "dashboard").glob("*.py"))}}
    (ROOT / "reports/step8-evidence/independent-apptest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
