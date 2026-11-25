"""Cell type annotation (CTA) figures from saved DeepSC test predictions.

Usage:
    python -m deepsc.viz.cta [--run-dir DIR] [--h5ad FILE] [--out DIR] [--class-names JSON]

Inputs:
  * ``<run-dir>/test_predictions.npz``: ``logits``, ``labels``, ``cell_ids``,
    ``classes`` and ``confusion_matrix`` for every test cell;
  * ``<run-dir>/status.json``: run record with test metrics and epoch history;
  * the test ``.h5ad`` the predictions were made on (raw counts, ``obs[celltype]``).

Outputs (in ``--out``): ``cta_annotation_comparison.png``, ``cta_annotation_flows.png``,
``cta_training_history.png``, ``cta_cells.csv`` and ``cta_per_class_metrics.csv``.

The UMAP is computed from the test cells' expression (TP10K, log1p, 2,000 HVGs,
50 PCs); it is a display layout, not a model embedding. Class codes are shown as
``Type <code>`` unless ``--class-names`` maps codes to names.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import argparse
from deepsc.viz._common import (
    DOWNSTREAM_RUN_ROOT,
    Checks,
    read_status,
    savefig,
    spread_labels,
    use_headless_backend,
)

DEFAULT_RUN_DIR = DOWNSTREAM_RUN_ROOT / "cta_zheng_1215"
DEFAULT_H5AD = Path("/DATA2/DeepSC/cell_type_annotation/zheng_test.h5ad")
CELLTYPE_COL = "celltype"
SEED = 0


@dataclass
class CTAResults:
    classes: np.ndarray  # class codes, index = model output index
    labels: np.ndarray  # true class index per cell
    logits: np.ndarray
    cell_ids: np.ndarray
    stored_confusion: np.ndarray
    status: dict
    adata: ad.AnnData

    @property
    def predictions(self) -> np.ndarray:
        return self.logits.argmax(1)


def load(run_dir: Path = DEFAULT_RUN_DIR, h5ad: Path = DEFAULT_H5AD) -> CTAResults:
    z = np.load(Path(run_dir) / "test_predictions.npz")
    return CTAResults(
        classes=z["classes"],
        labels=z["labels"],
        logits=z["logits"],
        cell_ids=z["cell_ids"],
        stored_confusion=z["confusion_matrix"],
        status=read_status(run_dir),
        adata=ad.read_h5ad(h5ad),
    )


def verify(r: CTAResults, celltype_col: str = CELLTYPE_COL) -> Checks:
    """Check alignment with the h5ad and that the recorded metrics are reproduced."""
    c = Checks()
    pred = r.predictions
    test = r.status["test"]
    c.add(
        f"cells {len(r.cell_ids)} == h5ad {r.adata.n_obs} == recorded {r.status['counts']['test']}",
        len(r.cell_ids) == r.adata.n_obs == r.status["counts"]["test"],
    )
    if len(r.cell_ids) == r.adata.n_obs:
        c.add(
            "cell order == h5ad order", (r.cell_ids == r.adata.obs_names.values).all()
        )
        c.add(
            "labels == h5ad cell types",
            (r.classes[r.labels] == r.adata.obs[celltype_col].astype(str).values).all(),
        )
    c.add("classes == recorded", list(r.classes) == r.status["classes"])
    c.add(
        "confusion matrix == stored",
        (
            confusion_matrix(r.labels, pred, labels=range(len(r.classes)))
            == r.stored_confusion
        ).all(),
    )
    c.close("accuracy", accuracy_score(r.labels, pred), test["accuracy"], atol=1e-9)
    c.close(
        "macro-F1",
        f1_score(r.labels, pred, average="macro"),
        test["macro_f1"],
        atol=1e-9,
    )
    return c


def class_order(
    classes: np.ndarray, names: dict[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Display name per code and display order (numeric when codes are integers)."""
    names = names or {}
    display = {code: names.get(code, f"Type {code}") for code in classes}
    try:
        codes = sorted(classes, key=int)
    except ValueError:
        codes = sorted(classes)
    return display, [display[code] for code in codes]


def expression_umap(adata: ad.AnnData, seed: int = SEED) -> np.ndarray:
    import scanpy as sc

    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a.obsm["X_umap"]


def cell_table(
    r: CTAResults, display: dict[str, str], umap: np.ndarray | None
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "cell_id": r.cell_ids,
            "ground_truth": [display[c] for c in r.classes[r.labels]],
            "prediction": [display[c] for c in r.classes[r.predictions]],
        }
    )
    if umap is not None:
        df["UMAP1"], df["UMAP2"] = umap[:, 0], umap[:, 1]
    return df


def per_class_metrics(
    df: pd.DataFrame, order: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    mat = confusion_matrix(df.ground_truth, df.prediction, labels=order)
    prec, rec, f1, _ = precision_recall_fscore_support(
        df.ground_truth, df.prediction, labels=order, zero_division=0
    )
    table = pd.DataFrame(
        {
            "support": mat.sum(1),
            "correct": np.diag(mat),
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        index=pd.Index(order, name="class"),
    )
    return mat, table


def palette(order: list[str]) -> dict:
    import seaborn as sns

    base = sns.color_palette("tab20", 20)
    picks = [0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 5, 14, 3, 7, 9, 11, 13, 15, 17, 19]
    return {lab: base[picks[i % len(picks)]] for i, lab in enumerate(order)}


def plot_overview(df, mat, metrics, order, colors, title_suffix, path: Path) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="notebook", font_scale=0.94)
    support = metrics.support.to_numpy()
    acc = (df.ground_truth == df.prediction).mean()
    fig = plt.figure(figsize=(20, 14.5), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.12])
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_num, ax_norm, ax_metric = (fig.add_subplot(gs[1, i]) for i in range(3))

    xl, yl = "UMAP 1 (expression)", "UMAP 2 (expression)"
    for ax, col, title in zip(
        top[:2],
        ["ground_truth", "prediction"],
        ["A · Ground truth", "B · DeepSC prediction"],
    ):
        for label in order:
            d = df[df[col] == label]
            ax.scatter(
                d.UMAP1,
                d.UMAP2,
                s=4,
                color=colors[label],
                alpha=0.8,
                label=label,
                linewidths=0,
                rasterized=True,
            )
        ax.set(title=title, xlabel=xl, ylabel=yl)
        ax.set_aspect("equal", adjustable="datalim")
    top[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=9,
        markerscale=3,
    )
    wrong = (df.ground_truth != df.prediction).to_numpy()
    top[2].scatter(
        df.UMAP1, df.UMAP2, s=3, c="#dce2e8", alpha=0.5, linewidths=0, rasterized=True
    )
    top[2].scatter(
        df.UMAP1[wrong],
        df.UMAP2[wrong],
        s=6,
        c="#d4464c",
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    top[2].set(
        title=f"C · Misclassified cells ({wrong.sum():,} / {len(df):,})",
        xlabel=xl,
        ylabel=yl,
    )
    top[2].set_aspect("equal", adjustable="datalim")

    ylab = [f"{name} (n={n})" for name, n in zip(order, support)]
    heat = dict(
        xticklabels=order,
        linewidths=0.3,
        linecolor="white",
        annot=True,
        annot_kws={"fontsize": 8},
    )
    sns.heatmap(
        mat, fmt="d", cmap="Blues", cbar=False, yticklabels=ylab, ax=ax_num, **heat
    )
    ax_num.set(title="D · Confusion counts", xlabel="Predicted", ylabel="Ground truth")
    with np.errstate(invalid="ignore", divide="ignore"):
        row_pct = 100 * mat / support[:, None]
    sns.heatmap(
        row_pct,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% of true class"},
        yticklabels=order,
        ax=ax_norm,
        **heat,
    )
    ax_norm.set(
        title="E · Row-normalized confusion (%)",
        xlabel="Predicted",
        ylabel="Ground truth",
    )
    for ax in (ax_num, ax_norm):
        ax.tick_params(axis="x", labelrotation=45)
        ax.tick_params(axis="y", labelrotation=0)

    yy, h = np.arange(len(order)), 0.26
    for off, col, color, lab in [
        (-h, "recall", "#3182bd", "Recall"),
        (0, "precision", "#f39c65", "Precision"),
        (h, "f1", "#48a78d", "F1"),
    ]:
        ax_metric.barh(yy + off, metrics[col], height=h, color=color, label=lab)
    for y, v, n_ok in zip(yy, metrics.f1, metrics.correct):
        ax_metric.text(
            v + 0.01, y + h, f"{v:.2f}", va="center", fontsize=8, color="#2d6e5c"
        )
        if n_ok == 0:
            ax_metric.text(
                0.01,
                y,
                "no correct predictions",
                va="center",
                fontsize=8,
                color="#d4464c",
            )
    ax_metric.set(
        yticks=yy,
        yticklabels=ylab,
        xlim=(0, 1.04),
        xlabel="Score",
        title="F · Per-class metrics (test set)",
    )
    ax_metric.invert_yaxis()
    ax_metric.legend(frameon=False, ncol=3, loc="lower right")
    ax_metric.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Cell type annotation · {title_suffix} (accuracy {acc:.2%}, macro-F1 {metrics.f1.mean():.3f})",
        fontsize=17,
        weight="bold",
    )
    return savefig(fig, path, dpi=175)


def plot_flows(mat, order, colors, title, path: Path) -> Path:
    """Sankey-style flows: ribbon width = cells; colored = correct, gray = mismatch."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    n, row, col = mat.sum(), mat.sum(1), mat.sum(0)
    left0, right0 = np.r_[0, np.cumsum(row[:-1])] / n, np.r_[0, np.cumsum(col[:-1])] / n
    used_l, used_r = np.zeros(len(order)), np.zeros(len(order))
    x0, x1, bend = 0.16, 0.84, 0.22
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    fig, ax = plt.subplots(figsize=(13, 8.5), layout="constrained")
    for i, j in zip(*np.nonzero(mat)):  # row-major order tiles each bar exactly
        h = mat[i, j] / n
        a, b = left0[i] + used_l[i], right0[j] + used_r[j]
        used_l[i] += h
        used_r[j] += h
        verts = [
            (x0, a),
            (x0 + bend, a),
            (x1 - bend, b),
            (x1, b),
            (x1, b + h),
            (x1 - bend, b + h),
            (x0 + bend, a + h),
            (x0, a + h),
            (x0, a),
        ]
        ok = i == j
        ax.add_patch(
            PathPatch(
                MplPath(verts, codes),
                fc=colors[order[i]] if ok else "#a5afb9",
                alpha=0.7 if ok else 0.4,
                lw=0,
            )
        )
    lab_l = spread_labels(left0 + row / (2 * n), gap=0.024)
    lab_r = spread_labels(right0 + col / (2 * n), gap=0.024)
    for k, name in enumerate(order):
        ax.plot(
            [0.15, 0.15],
            [left0[k], left0[k] + row[k] / n],
            color=colors[name],
            lw=12,
            solid_capstyle="butt",
        )
        ax.plot(
            [0.85, 0.85],
            [right0[k], right0[k] + col[k] / n],
            color=colors[name],
            lw=12,
            solid_capstyle="butt",
        )
        ax.text(
            0.125, lab_l[k], f"{name}  {row[k]}", ha="right", va="center", fontsize=9
        )
        ax.text(
            0.875, lab_r[k], f"{name}  {col[k]}", ha="left", va="center", fontsize=9
        )
    ax.set(xlim=(-0.12, 1.15), ylim=(1.015, -0.015))
    ax.set_title(title, pad=30)
    ax.text(0.15, -0.04, "GROUND TRUTH", ha="center", weight="bold")
    ax.text(0.85, -0.04, "PREDICTION", ha="center", weight="bold")
    ax.text(
        0.5,
        1.035,
        "Ribbon width = cell count · colored: correct · gray: mismatch",
        ha="center",
        va="top",
        fontsize=10,
        color="#56616c",
    )
    ax.axis("off")
    return savefig(fig, path, dpi=175)


def plot_history(status: dict, title: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    h, best = status["history"], status["best_epoch"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    ax[0].plot(ep, [r["train_loss"] for r in h], "o-", label="Train loss")
    ax[0].plot(ep, [r["val"]["loss"] for r in h], "o-", label="Validation loss")
    ax[1].plot(ep, [r["val"]["accuracy"] for r in h], "o-", label="Validation accuracy")
    ax[1].plot(ep, [r["val"]["macro_f1"] for r in h], "o-", label="Validation macro-F1")
    for a, ylabel in zip(ax, ["Cross-entropy loss", "Score"]):
        a.axvline(best, ls="--", c="#888", lw=1)
        a.set(xlabel="Epoch", ylabel=ylabel, xticks=ep)
        a.legend(frameon=False)
        a.grid(alpha=0.25)
    ax[0].set_title(f"A · Loss (best checkpoint = epoch {best}, min val loss)")
    ax[1].set_title("B · Validation metrics")
    fig.suptitle(title, fontsize=14, weight="bold")
    return savefig(fig, path, dpi=160)


def run(
    run_dir: Path,
    h5ad: Path,
    out: Path,
    class_names: dict[str, str] | None = None,
    dataset_label: str = "Zheng",
) -> pd.DataFrame:
    """Verify, then write all CTA figures and tables to ``out``. Returns per-class metrics."""
    out.mkdir(parents=True, exist_ok=True)
    r = load(run_dir, h5ad)
    print("Verification:")
    checks = verify(r)
    checks.report()
    checks.require()

    display, order = class_order(r.classes, class_names)
    df = cell_table(r, display, expression_umap(r.adata))
    mat, metrics = per_class_metrics(df, order)
    # The display-ordered matrix must be a permutation of the stored one.
    perm = sorted(
        range(len(r.classes)), key=lambda k: order.index(display[r.classes[k]])
    )
    Checks(
        {
            "plotted confusion matrix == stored (reordered)": bool(
                (mat == r.stored_confusion[np.ix_(perm, perm)]).all()
            )
        }
    ).report()

    colors = palette(order)
    plot_overview(
        df,
        mat,
        metrics,
        order,
        colors,
        f"DeepSC fine-tuned on {dataset_label} · test set",
        out / "cta_annotation_comparison.png",
    )
    plot_flows(
        mat,
        order,
        colors,
        f"Ground truth → DeepSC prediction · {dataset_label} test cells",
        out / "cta_annotation_flows.png",
    )
    plot_history(
        r.status,
        f"CTA fine-tuning history · {dataset_label}",
        out / "cta_training_history.png",
    )
    df.to_csv(out / "cta_cells.csv", index=False)
    metrics.round(4).to_csv(out / "cta_per_class_metrics.csv")
    return metrics


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="dir with test_predictions.npz, status.json",
    )
    p.add_argument(
        "--h5ad",
        type=Path,
        default=DEFAULT_H5AD,
        help="test h5ad the predictions were made on",
    )
    p.add_argument("--out", type=Path, default=Path("viz_outputs/cta"))
    p.add_argument(
        "--class-names", type=Path, help="optional JSON {code: name} for display"
    )
    p.add_argument("--dataset-label", default="Zheng")
    args = p.parse_args(argv)
    use_headless_backend()
    names = json.loads(args.class_names.read_text()) if args.class_names else None
    metrics = run(args.run_dir, args.h5ad, args.out, names, args.dataset_label)
    print(metrics.round(3).to_string())
    print("Saved to", args.out.resolve())


if __name__ == "__main__":
    main()
