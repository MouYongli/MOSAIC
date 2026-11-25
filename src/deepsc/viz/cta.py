"""Cell type annotation (CTA) figures for one prediction set.

Usage (normally via ``deepsc report``):
    python -m deepsc.viz.cta PRED_DIR [--dataset zheng|PATH.yaml] [--out DIR]

Inputs:
  * a CTA prediction set (``deepsc.pipeline.contract``): ``cell_ids``,
    ``classes``, ``y_true``, ``logits`` and ``meta.json``;
  * the dataset config ``configs/datasets/cta/<dataset>.yaml``: h5ad path per
    split, label column and the code -> cell type name mapping (``classes``);
  * optionally the model's fine-tuning run record (``training_record`` in the
    model config) for the training-history figures.

Writes ``<out>/cta/<dataset>/<model>/`` (default ``outputs/viz``; see
:mod:`deepsc.viz._common` for the layout): one untitled figure per panel, each
with a ``.json`` sidecar holding its title, short and long caption and a text
description.

The UMAP is computed from the cells' expression (TP10K, log1p, 2,000 HVGs,
50 PCs); it is a display layout, not a model embedding. Class codes are shown as
``Type <code>`` when the dataset config has no ``classes`` mapping.
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
from deepsc.pipeline.contract import PredictionSet
from deepsc.viz._common import (
    DEFAULT_OUT,
    Checks,
    FigureWriter,
    apply_style,
    load_dataset_config,
    spread_labels,
    training_record,
    use_headless_backend,
)

CELLTYPE_COL = "celltype"
SEED = 0
UMAP_METHOD = (
    "computed from expression (library size normalized to 10,000 counts, log1p, 2,000 highly variable "
    "genes, 50 principal components, 15 nearest neighbours)"
)

# --------------------------------------------------------------------------- data


@dataclass
class CTAResults:
    classes: np.ndarray  # class codes, index = model output index
    labels: np.ndarray  # true class index per cell (-1 = not a model class)
    logits: np.ndarray
    cell_ids: np.ndarray
    adata: ad.AnnData
    meta: dict
    training: dict | None = None  # fine-tuning run record (status.json), if known

    @property
    def predictions(self) -> np.ndarray:
        return self.logits.argmax(1)

    @property
    def model_id(self) -> str:
        return self.meta["model"]["id"]

    @property
    def recorded_test(self) -> dict | None:
        """Test metrics recorded by an imported legacy run (same predictions, so must match)."""
        return (self.meta.get("legacy") or {}).get("recorded_test")


def load(pred_dir: Path, dataset: dict) -> CTAResults:
    ps = PredictionSet.load(pred_dir)
    if ps.task != "cta":
        raise ValueError(f"{pred_dir} holds {ps.task!r} predictions, not 'cta'")
    a = ps.arrays
    return CTAResults(
        classes=a["classes"],
        labels=a["y_true"],
        logits=a["logits"],
        cell_ids=a["cell_ids"],
        adata=ad.read_h5ad(dataset["data"][ps.meta["split"]]),
        meta=ps.meta,
        training=training_record(ps.meta),
    )


def verify(r: CTAResults, celltype_col: str = CELLTYPE_COL) -> Checks:
    """Check alignment with the h5ad and, for legacy runs, the recorded metrics."""
    c = Checks()
    n = len(r.cell_ids)
    c.add(f"cells {n} == h5ad {r.adata.n_obs}", n == r.adata.n_obs)
    if n == r.adata.n_obs:
        c.add(
            "cell order == h5ad order", (r.cell_ids == r.adata.obs_names.values).all()
        )
        known = r.labels >= 0
        c.add(
            "labels == h5ad cell types",
            (
                r.classes[r.labels[known]]
                == r.adata.obs[celltype_col].astype(str).values[known]
            ).all(),
        )
    c.add("every cell's label is a model class", (r.labels >= 0).all())
    if r.recorded_test:
        pred, test = r.predictions, r.recorded_test
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


def top_confusions(mat: np.ndarray, order: list[str], k: int = 5) -> list[dict]:
    """Largest off-diagonal cells: true class, predicted class, count, % of the true class."""
    off = mat.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [
        {
            "true": order[i],
            "predicted": order[j],
            "cells": int(off[i, j]),
            "pct_of_true": 100 * off[i, j] / mat[i].sum(),
        }
        for i, j in idx
        if off[i, j] > 0
    ]


def palette(order: list[str]) -> dict:
    import seaborn as sns

    base = sns.color_palette("tab20", 20)
    picks = [0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 5, 14, 3, 7, 9, 11, 13, 15, 17, 19]
    return {lab: base[picks[i % len(picks)]] for i, lab in enumerate(order)}


def _fmt_list(items: list[str]) -> str:
    return (
        ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]
        if items
        else "none"
    )


# ------------------------------------------------------------------------ figures


def _umap_axes(ax):
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)


def fig_umap_labels(df, col, order, colors):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5))
    for label in order:
        d = df[df[col] == label]
        ax.scatter(
            d.UMAP1,
            d.UMAP2,
            s=3,
            color=colors[label],
            alpha=0.8,
            label=label,
            linewidths=0,
            rasterized=True,
        )
    _umap_axes(ax)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        markerscale=4,
        fontsize=8,
    )
    return fig


def fig_umap_errors(df):
    import matplotlib.pyplot as plt

    wrong = (df.ground_truth != df.prediction).to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        df.UMAP1,
        df.UMAP2,
        s=3,
        c="#dce2e8",
        alpha=0.6,
        linewidths=0,
        rasterized=True,
        label="Correct",
    )
    ax.scatter(
        df.UMAP1[wrong],
        df.UMAP2[wrong],
        s=5,
        c="#d4464c",
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        label="Misclassified",
    )
    _umap_axes(ax)
    ax.legend(loc="lower left", frameon=False, markerscale=4, fontsize=8)
    return fig


def fig_confusion(mat, order, normalized: bool):
    import matplotlib.pyplot as plt
    import seaborn as sns

    support = mat.sum(1)
    fig, ax = plt.subplots(figsize=(7, 6))
    kw = dict(
        xticklabels=order,
        linewidths=0.3,
        linecolor="white",
        annot=True,
        annot_kws={"fontsize": 7},
        ax=ax,
    )
    if normalized:
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = 100 * mat / support[:, None]
        sns.heatmap(
            pct,
            fmt=".0f",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "% of true class"},
            yticklabels=order,
            **kw,
        )
    else:
        sns.heatmap(
            mat,
            fmt="d",
            cmap="Blues",
            cbar=False,
            yticklabels=[f"{n} (n={s})" for n, s in zip(order, support)],
            **kw,
        )
    ax.set(xlabel="Predicted", ylabel="Ground truth")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelrotation=0)
    return fig


def fig_per_class(metrics: pd.DataFrame):
    import matplotlib.pyplot as plt

    order = list(metrics.index)
    yy, h = np.arange(len(order)), 0.26
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for off, col, color, lab in [
        (-h, "recall", "#3182bd", "Recall"),
        (0, "precision", "#f39c65", "Precision"),
        (h, "f1", "#48a78d", "F1"),
    ]:
        ax.barh(yy + off, metrics[col], height=h, color=color, label=lab)
    for y, v, n_ok in zip(yy, metrics.f1, metrics.correct):
        if n_ok > 0:
            ax.text(
                v + 0.01, y + h, f"{v:.2f}", va="center", fontsize=7, color="#2d6e5c"
            )
        else:
            ax.text(
                0.01,
                y,
                "no correct predictions",
                va="center",
                fontsize=7,
                color="#d4464c",
            )
    ax.set(
        yticks=yy,
        yticklabels=[f"{n} (n={s})" for n, s in zip(order, metrics.support)],
        xlim=(0, 1.06),
        xlabel="Score",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", visible=False)
    return fig


def fig_flows(mat, order, colors):
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
    fig, ax = plt.subplots(figsize=(9, 7))
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
    lab_l = spread_labels(left0 + row / (2 * n), gap=0.026)
    lab_r = spread_labels(right0 + col / (2 * n), gap=0.026)
    for k, name in enumerate(order):
        ax.plot(
            [0.15, 0.15],
            [left0[k], left0[k] + row[k] / n],
            color=colors[name],
            lw=10,
            solid_capstyle="butt",
        )
        ax.plot(
            [0.85, 0.85],
            [right0[k], right0[k] + col[k] / n],
            color=colors[name],
            lw=10,
            solid_capstyle="butt",
        )
        ax.text(
            0.125, lab_l[k], f"{name}  {row[k]}", ha="right", va="center", fontsize=8
        )
        ax.text(
            0.875, lab_r[k], f"{name}  {col[k]}", ha="left", va="center", fontsize=8
        )
    ax.set(xlim=(-0.12, 1.15), ylim=(1.015, -0.05))
    ax.text(0.15, -0.03, "Ground truth", ha="center", weight="bold")
    ax.text(0.85, -0.03, "Prediction", ha="center", weight="bold")
    ax.axis("off")
    return fig


def fig_history(status: dict, which: str):
    import matplotlib.pyplot as plt

    h, best = status["history"], status["best_epoch"]
    ep = [x["epoch"] for x in h]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    if which == "loss":
        ax.plot(ep, [x["train_loss"] for x in h], "o-", label="Training")
        ax.plot(ep, [x["val"]["loss"] for x in h], "o-", label="Validation")
        ax.set_ylabel("Cross-entropy loss")
    else:
        ax.plot(ep, [x["val"]["accuracy"] for x in h], "o-", label="Accuracy")
        ax.plot(ep, [x["val"]["macro_f1"] for x in h], "o-", label="Macro-F1")
        ax.set_ylabel("Validation score")
    ax.axvline(
        best, ls="--", c="#888", lw=1, label=f"Selected checkpoint (epoch {best})"
    )
    ax.set(xlabel="Epoch", xticks=ep)
    ax.legend(frameon=False, fontsize=8)
    return fig


# ----------------------------------------------------------------------- pipeline


def class_names_from_config(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """(code -> short display name, short -> full name) from a dataset config's ``classes``."""
    classes = cfg.get("classes") or {}
    short = {str(code): c.get("short") or c["name"] for code, c in classes.items()}
    full = {short[str(code)]: c["name"] for code, c in classes.items()}
    return short, full


def run(pred_dir: Path, out: Path, dataset: dict) -> FigureWriter:
    """Verify, then write all CTA figures, metadata and tables for one prediction set.

    ``pred_dir`` is a prediction set (see ``deepsc.pipeline.contract``) and
    ``dataset`` its dataset config. Output: ``<out>/cta/<dataset>/<model>/``.
    """
    class_names, full_names = class_names_from_config(dataset)
    dataset_label = dataset.get("label", dataset["name"])
    r = load(pred_dir, dataset)
    print("Verification:")
    checks = verify(r, dataset.get("celltype_col", CELLTYPE_COL))
    if class_names:
        checks.add(
            "dataset config names every class code",
            set(map(str, r.classes)) <= set(class_names),
        )
    checks.report()
    checks.require()

    display, order = class_order(r.classes, class_names)
    df = cell_table(r, display, expression_umap(r.adata))
    mat, metrics = per_class_metrics(df, order)
    perm = sorted(
        range(len(r.classes)), key=lambda k: order.index(display[r.classes[k]])
    )
    stored = confusion_matrix(r.labels, r.predictions, labels=range(len(r.classes)))
    checks.add(
        "plotted confusion matrix == predictions (display order)",
        (mat == stored[np.ix_(perm, perm)]).all(),
    )
    checks.require()

    apply_style()
    colors = palette(order)
    w = FigureWriter(
        out,
        "cta",
        dataset["name"],
        dataset_label,
        {
            "predictions": Path(pred_dir).resolve(),
            "h5ad": dataset["data"][r.meta["split"]],
            "dataset_config": dataset["_path"],
            "model_config": r.meta["model"].get("config", ""),
        },
        model=r.model_id,
    )
    ds, n, k = dataset_label, len(df), len(order)
    acc, mf1 = float((df.ground_truth == df.prediction).mean()), float(
        metrics.f1.mean()
    )
    n_wrong = int((df.ground_truth != df.prediction).sum())
    conf = top_confusions(mat, order)
    size_diff = np.argsort(-np.abs(mat.sum(0) - mat.sum(1)))[:3]
    size_txt = _fmt_list(
        [
            f"{order[i]} ({mat.sum(1)[i]} reference vs {mat.sum(0)[i]} predicted)"
            for i in size_diff
        ]
    )
    top_share = sum(c["cells"] for c in conf) / max(n_wrong, 1)
    conf_txt = "; ".join(
        f"{c['true']} → {c['predicted']}: {c['cells']} cells ({c['pct_of_true']:.0f}%)"
        for c in conf[:4]
    )
    low = list(metrics.index[metrics.recall < 0.5])
    high = list(metrics.index[metrics.f1 >= 0.85])
    zero = list(metrics.index[metrics.correct == 0])
    abbrev = [
        f"{s_} = {f_}" for s_, f_ in full_names.items() if s_ in order and s_ != f_
    ]
    names_note = (
        "Cell types are shown by their dataset codes because no names are configured."
        if not class_names
        else (f"Abbreviations: {'; '.join(abbrev)}." if abbrev else "")
    )
    sel = (
        f"Model: {r.model_id}; the checkpoint with the lowest validation loss (epoch "
        f"{r.training['best_epoch']}) was evaluated."
        if r.training
        else f"Model: {r.model_id}."
    )
    base = {"cells": n, "classes": k, "accuracy": acc, "macro_f1": mf1}

    w.save(
        fig_umap_labels(df, "ground_truth", order, colors),
        "umap_ground_truth",
        title=f"{ds} test cells colored by reference annotation",
        caption_short=f"Reference cell type annotation of the {ds} test set.",
        caption_long=f"UMAP of {n:,} {ds} test cells, {UMAP_METHOD}, colored by the reference annotation "
        f"({k} cell types). {names_note}".strip(),
        description="Each dot is one test cell; position reflects expression similarity only and is identical "
        "across the three UMAP panels, so the reference labels here can be compared directly with "
        "the DeepSC predictions and the error map. Separated islands correspond to distinct cell "
        "populations; populations that overlap in this layout are expected to be harder to tell "
        "apart.",
        metrics=base,
    )
    w.save(
        fig_umap_labels(df, "prediction", order, colors),
        "umap_prediction",
        title=f"{ds} test cells colored by DeepSC prediction",
        caption_short=f"DeepSC cell type predictions on the {ds} test set.",
        caption_long=f"Same UMAP layout as the reference panel, colored by the cell type predicted by fine-tuned "
        f"DeepSC (argmax of the classifier logits). Overall accuracy {acc:.1%}, macro-F1 {mf1:.3f}. "
        f"{sel} {names_note}".strip(),
        description=f"Compare with the reference panel cell by cell: same layout, different labels. "
        f"{acc:.1%} of cells receive their reference label. Predicted class sizes differ most from "
        f"the reference for {size_txt}.",
        metrics={**base, "predicted_class_sizes": dict(zip(order, mat.sum(0)))},
    )
    w.save(
        fig_umap_errors(df),
        "umap_errors",
        title=f"Misclassified {ds} test cells",
        caption_short=f"Location of misclassified cells ({n_wrong:,} of {n:,}).",
        caption_long=f"Same UMAP layout; cells whose DeepSC prediction differs from the reference annotation are "
        f"shown in red ({n_wrong:,} of {n:,} cells, {n_wrong / n:.1%}), correctly classified cells "
        f"in gray.",
        description=f"Red cells mark where the classifier disagrees with the reference. The {len(conf)} largest "
        f"class-to-class confusions account for {top_share:.0%} of all errors, i.e. errors are "
        f"{'concentrated in a few' if top_share >= 0.5 else 'spread over many'} pairs of cell types; "
        f"the largest are {conf_txt}. Compare with the reference panel to see whether these pairs "
        f"overlap in expression space.",
        metrics={**base, "misclassified": n_wrong, "top_confusions": conf},
    )
    w.save(
        fig_confusion(mat, order, normalized=False),
        "confusion_counts",
        title=f"Confusion matrix (cell counts), {ds}",
        caption_short=f"Confusion matrix of DeepSC on the {ds} test set (counts).",
        caption_long=f"Rows: reference cell type (with number of test cells); columns: predicted cell type; "
        f"entries: number of cells. Diagonal entries are correct predictions ({n - n_wrong:,} of "
        f"{n:,}).",
        description=f"Read along a row to see where the cells of one reference type were assigned. Large "
        f"off-diagonal entries mark systematic confusions: {conf_txt}. Class sizes are highly "
        f"imbalanced (from {metrics.support.min()} to {metrics.support.max():,} cells), so counts are "
        f"dominated by the largest classes; see the row-normalized matrix for per-class rates.",
        metrics={**base, "confusion_matrix": mat, "order": order},
    )
    w.save(
        fig_confusion(mat, order, normalized=True),
        "confusion_row_normalized",
        title=f"Row-normalized confusion matrix, {ds}",
        caption_short=f"Per-class assignment rates of DeepSC on the {ds} test set.",
        caption_long="Each row is normalized to 100% of the cells of that reference type; the diagonal equals "
        "per-class recall. Rows: reference cell type; columns: predicted cell type.",
        description=f"The diagonal shows how often each type is recovered. Recall is below 50% for "
        f"{_fmt_list(low)}; {_fmt_list(zero)} had no correctly classified cells. Off-diagonal mass "
        f"shows where misassigned cells go, e.g. {conf_txt}.",
        metrics={**base, "recall": dict(zip(order, metrics.recall))},
    )
    w.save(
        fig_per_class(metrics),
        "per_class_metrics",
        title=f"Per-class precision, recall and F1, {ds}",
        caption_short=f"Per-class classification metrics on the {ds} test set.",
        caption_long=f"Recall, precision and F1 for each reference cell type (number of test cells in "
        f"parentheses). Macro-F1 ({mf1:.3f}) is the unweighted mean of the per-class F1 scores; "
        f"accuracy is {acc:.1%}.",
        description=f"F1 is at least 0.85 for {_fmt_list(high)}. The gap between accuracy ({acc:.1%}) and "
        f"macro-F1 ({mf1:.3f}) comes from small or ambiguous classes with low recall "
        f"({_fmt_list(low)}), which weigh equally in the macro average.",
        metrics={**base, "per_class": metrics.round(4).to_dict(orient="index")},
    )
    w.save(
        fig_flows(mat, order, colors),
        "flows",
        title=f"Reference-to-prediction flows, {ds}",
        caption_short=f"Flow of {ds} test cells from reference to predicted cell types.",
        caption_long="Left: reference cell types; right: predicted cell types; bar heights and ribbon widths are "
        "proportional to cell counts. Colored ribbons are correct assignments, gray ribbons are "
        "misassignments.",
        description="An alternative view of the confusion matrix that emphasizes how large each misassignment is "
        "relative to the class sizes, and how predicted class sizes differ from the reference. It "
        "carries the same information as the confusion matrix and is suited to supplementary "
        "material.",
        metrics={
            **base,
            "reference_sizes": dict(zip(order, mat.sum(1))),
            "predicted_sizes": dict(zip(order, mat.sum(0))),
        },
    )

    if r.training:
        _save_history(w, r.training, ds)
    w.table(df, "cells", index=False)
    w.table(metrics.round(4), "per_class_metrics")
    w.finalize(
        checks,
        summary={
            **base,
            "model": r.model_id,
            "split": r.meta["split"],
            "misclassified": n_wrong,
            "weighted_f1": float(
                f1_score(df.ground_truth, df.prediction, average="weighted")
            ),
            "per_class_f1": metrics.f1.round(4).to_dict(),
        },
    )
    return w


def _save_history(w: FigureWriter, training: dict, ds: str) -> None:
    h = training["history"]
    best = training["best_epoch"]
    bv = next(x["val"] for x in h if x["epoch"] == best)
    last = h[-1]
    mono = all(b["train_loss"] < a["train_loss"] for a, b in zip(h, h[1:]))
    best_f1 = max(h, key=lambda x: x["val"]["macro_f1"])
    w.save(
        fig_history(training, "loss"),
        "training_loss",
        title=f"CTA fine-tuning loss, {ds}",
        caption_short=f"Training and validation loss during CTA fine-tuning on {ds}.",
        caption_long=f"Cross-entropy loss per epoch on the training and validation sets. The dashed line marks "
        f"the checkpoint with the lowest validation loss (epoch {best}), which was used for all test "
        f"results. Training stopped after epoch {last['epoch']} (early stopping).",
        description=f"Training loss {'decreases monotonically' if mono else 'decreases'} from "
        f"{h[0]['train_loss']:.3f} to {last['train_loss']:.3f}. Validation loss reaches its minimum "
        f"({bv['loss']:.3f}) at epoch {best}"
        + (
            f" and ends at {last['val']['loss']:.3f} (epoch {last['epoch']}); the widening gap to the "
            f"training loss indicates the onset of overfitting."
            if last["epoch"] > best
            else "."
        ),
        metrics={"history": h, "best_epoch": best},
    )
    w.save(
        fig_history(training, "val"),
        "training_validation_metrics",
        title=f"CTA validation accuracy and macro-F1, {ds}",
        caption_short=f"Validation accuracy and macro-F1 during CTA fine-tuning on {ds}.",
        caption_long=f"Accuracy and macro-F1 on the validation set ({bv['cells']:,} cells) after each epoch; "
        f"the dashed line marks the selected checkpoint (epoch {best}, lowest validation loss).",
        description=f"At the selected checkpoint validation accuracy is {bv['accuracy']:.1%} and macro-F1 "
        f"{bv['macro_f1']:.3f}. Best validation macro-F1 is {best_f1['val']['macro_f1']:.3f} at epoch "
        f"{best_f1['epoch']}"
        + (
            "; because the checkpoint is selected by loss rather than macro-F1, the selection criterion "
            "affects the reported test macro-F1 and should be stated."
            if best_f1["epoch"] != best
            else ", the same epoch as the selected checkpoint."
        ),
        metrics={"history": h, "best_epoch": best},
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "predictions",
        type=Path,
        help="prediction set dir (predictions.npz + meta.json)",
    )
    p.add_argument(
        "--dataset", help="dataset config name or YAML path (default: from meta.json)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="root dir; writes <out>/cta/<dataset>/<model>/",
    )
    args = p.parse_args(argv)
    use_headless_backend()
    dataset = (
        args.dataset
        or json.loads((args.predictions / "meta.json").read_text())["dataset"]
    )
    w = run(args.predictions, args.out, load_dataset_config("cta", dataset))
    print(f"Saved {len(w.records)} figures to {w.root.resolve()}")


if __name__ == "__main__":
    main()
