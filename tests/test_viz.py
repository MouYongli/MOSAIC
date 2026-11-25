"""Unit tests for deepsc.viz on small synthetic data (no /DATA2 access needed)."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from deepsc.viz import cta
from deepsc.viz import perturbation as pp
from deepsc.viz._common import Checks, VerificationError, spread_labels


def test_spread_labels_enforces_gap_and_keeps_order():
    y = spread_labels(np.array([0.10, 0.11, 0.115, 0.5]), gap=0.02)
    assert np.all(np.diff(y) >= 0.02 - 1e-12)
    assert y[0] == 0.10 and y[-1] == 0.5


def test_checks_require_raises_on_failure():
    c = Checks()
    c.add("ok", True)
    c.require()
    c.add("bad", False)
    with pytest.raises(VerificationError, match="bad"):
        c.require()


# ----------------------------------------------------------------------------- CTA


def _cta_results(corrupt_order=False):
    rng = np.random.default_rng(0)
    classes = np.array(["0", "1", "10", "2"])  # lexicographic, like the real run
    labels = rng.integers(0, 4, 40)
    logits = rng.normal(size=(40, 4))
    logits[np.arange(40), labels] += 2.0
    pred = logits.argmax(1)
    ids = np.array([f"Cell_{i}" for i in range(40)])
    obs = pd.DataFrame(
        {"celltype": classes[labels]}, index=ids[::-1] if corrupt_order else ids
    )
    adata = ad.AnnData(X=np.zeros((40, 3), dtype=np.float32), obs=obs)
    from sklearn.metrics import confusion_matrix, f1_score

    status = {
        "counts": {"test": 40},
        "classes": list(classes),
        "test": {
            "accuracy": float((pred == labels).mean()),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
        },
    }
    return cta.CTAResults(
        classes,
        labels,
        logits,
        ids,
        confusion_matrix(labels, pred, labels=range(4)),
        status,
        adata,
    )


def test_cta_verify_passes_on_consistent_data():
    cta.verify(_cta_results()).require()


def test_cta_verify_detects_misaligned_cells():
    with pytest.raises(VerificationError, match="cell order"):
        cta.verify(_cta_results(corrupt_order=True)).require()


def test_cta_class_order_is_numeric_and_metrics_match_confusion():
    r = _cta_results()
    display, order = cta.class_order(r.classes)
    assert order == ["Type 0", "Type 1", "Type 2", "Type 10"]
    mat, metrics = cta.per_class_metrics(cta.cell_table(r, display, None), order)
    assert mat.sum() == 40 and (metrics.support.to_numpy() == mat.sum(1)).all()
    assert (metrics.correct.to_numpy() == np.diag(mat)).all()


# ------------------------------------------------------------------------------ PP


def _pp_results(perfect=True):
    """Two test conditions + one train condition + ctrl; 6 genes, gene 5 unmapped."""
    rng = np.random.default_rng(1)
    n_gene = 6
    conds = ["ctrl"] * 30 + ["A+ctrl"] * 10 + ["B+ctrl"] * 10 + ["T+ctrl"] * 10
    X = rng.gamma(2.0, 1.0, size=(len(conds), n_gene)).astype(np.float32)
    X[30:40, :3] += 2.0
    X[40:50, 2:5] -= 1.0
    X = np.clip(X, 0, None)
    genes = np.array([f"G{i}" for i in range(n_gene)])
    obs = pd.DataFrame(
        {"condition": conds, "condition_name": [f"K562_{c}_1+1" for c in conds]}
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame({"gene_name": genes}, index=genes)
    )
    adata.uns["rank_genes_groups_cov_all"] = {
        f"K562_{c}_1+1": list(genes[[0, 1, 2, 5]]) for c in set(conds)
    }
    mapped = np.array([True] * 5 + [False])
    test = np.arange(30, 50)
    target = X[test].copy()
    target[:, ~mapped] = 0
    pred = (
        target.copy()
        if perfect
        else target + rng.normal(0, 0.3, target.shape).astype(np.float32)
    )
    pred[:, ~mapped] = 0
    status = {
        "counts": {"test": len(test)},
        "mapped_genes": int(mapped.sum()),
        "test": {},
    }
    r = pp.PPResults(
        pred, target, np.array(conds)[test], genes, mapped, status, adata, ["T+ctrl"]
    )
    return r, X


def test_pp_means_and_metrics_on_perfect_predictions():
    r, X = _pp_results(perfect=True)
    m = pp.compute_means(r)
    np.testing.assert_allclose(m.ctrl, X[:30].mean(0), rtol=1e-6)
    np.testing.assert_allclose(
        m.train_mean_delta, X[50:].mean(0) - X[:30].mean(0), rtol=1e-5
    )
    rec = pp.recorded_convention_metrics(r, m)
    assert rec["mse"] == 0 and rec["pearson_delta"] == pytest.approx(1.0)
    pc = pp.per_condition_metrics(r, m)
    assert (pc.n_de_mapped == 3).all()  # gene 5 is DE but unmapped -> excluded
    assert np.allclose(pc.pearson_delta_mapped, 1.0) and np.allclose(pc.mse_model, 0.0)


def test_pp_verify_reproduces_recorded_metrics_and_flags_mismatch():
    r, _ = _pp_results(perfect=False)
    m = pp.compute_means(r)
    r.status["test"] = pp.recorded_convention_metrics(r, m)
    pp.verify(r, m).require()
    r.status["test"]["pearson_delta"] += 0.01
    with pytest.raises(VerificationError, match="pearson_delta"):
        pp.verify(r, m).require()


def test_mean_pairwise_r_identical_rows_is_one():
    row = np.array([1.0, -2.0, 0.5, 3.0])
    assert pp.mean_pairwise_r(np.vstack([row, 2 * row, row + 1])) == pytest.approx(1.0)


def test_figure_writer_writes_untitled_figure_metadata_and_index(tmp_path):
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deepsc.viz._common import FigureWriter

    w = FigureWriter(
        tmp_path, "cta", "Zheng 68k", "Zheng 68k", {"h5ad": tmp_path / "x.h5ad"}
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    w.save(
        fig,
        "demo",
        title="T",
        caption_short="S.",
        caption_long="L.",
        description="D.",
        metrics={"r": np.float32(1)},
    )
    checks = Checks()
    checks.add("ok", True)
    w.finalize(checks, summary={"n": np.int64(3)})

    root = tmp_path / "cta" / "zheng_68k"
    meta = json.loads((root / "figures" / "cta_zheng_68k_demo.json").read_text())
    assert {
        meta[k] for k in ("title", "caption_short", "caption_long", "description")
    } == {"T", "S.", "L.", "D."}
    assert all((root / f).exists() for f in meta["files"].values())
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["figures"][0]["id"] == "cta_zheng_68k_demo" and manifest[
        "verification"
    ] == {"ok": True}
    assert "## 1. T" in (root / "README.md").read_text()
