"""Perturbation prediction (PP) figures for one prediction set.

Usage (normally via ``deepsc report``):
    python -m deepsc.viz.perturbation PRED_DIR [--dataset adamson|PATH.yaml] [--out DIR]

Inputs:
  * a PP prediction set (``deepsc.pipeline.contract``): per test cell
    ``predictions`` and ``targets`` (cells x genes), ``conditions``, ``genes``,
    ``gene_mask`` and ``meta.json``;
  * the dataset config ``configs/datasets/pp/<dataset>.yaml``: the GEARS
    ``perturb_processed.h5ad`` (control cells, ``uns[rank_genes_groups_cov_all]``)
    and the split pickle (training perturbations, for the train-mean baseline);
  * optionally the model's fine-tuning run record (training-history figures).

Writes ``<out>/pp/<dataset>/<model>/`` (default ``outputs/viz``; see
:mod:`deepsc.viz._common` for the layout): one untitled figure per panel, each
with a ``.json`` sidecar holding its title, short and long caption and a text
description.

All metrics and figures use the genes in ``gene_mask`` only. Imported legacy
runs (2026-11-20) zeroed unmapped genes in predictions AND targets and scored
all genes, which inflates delta correlations; for those, verification
reproduces the recorded metrics with that convention.

Baseline "train-mean Δ": predict the average Δ of all training perturbations for
every test perturbation (no learning involved).
"""

from __future__ import annotations
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import argparse
from deepsc.pipeline.contract import PredictionSet
from deepsc.viz._common import (
    DEFAULT_OUT,
    Checks,
    FigureWriter,
    apply_style,
    load_dataset_config,
    training_record,
    use_headless_backend,
)

CTRL = "ctrl"
N_DE = 20
SEED = 0


# --------------------------------------------------------------------------- data


@dataclass
class PPResults:
    pred: np.ndarray  # cells x genes
    target: np.ndarray  # cells x genes
    cond: np.ndarray  # condition per test cell
    genes: np.ndarray
    mapped: np.ndarray  # bool per gene: the model can represent it (gene_mask)
    adata: ad.AnnData
    train_conditions: list[str]
    meta: dict
    training: dict | None = None  # fine-tuning run record (status.json), if known

    @property
    def conditions(self) -> np.ndarray:
        return np.unique(self.cond)

    @property
    def model_id(self) -> str:
        return self.meta["model"]["id"]

    @property
    def recorded_test(self) -> dict | None:
        """Test metrics recorded by an imported legacy run (same predictions, so must match)."""
        return (self.meta.get("legacy") or {}).get("recorded_test")


@dataclass
class Means:
    """Per-condition mean profiles, computed once and shared by checks and plots."""

    pred: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    h5ad: dict[str, np.ndarray]  # same test conditions, straight from the h5ad
    ctrl: np.ndarray
    train_mean_delta: np.ndarray
    de: dict[str, np.ndarray]  # top-N DE gene indices per test condition


def load(pred_dir: Path, dataset: dict) -> PPResults:
    ps = PredictionSet.load(pred_dir)
    if ps.task != "pp":
        raise ValueError(f"{pred_dir} holds {ps.task!r} predictions, not 'pp'")
    with open(dataset["data"]["split"], "rb") as f:
        train = [c for c in pickle.load(f)["train"] if c != CTRL]
    a = ps.arrays
    return PPResults(
        pred=a["predictions"],
        target=a["targets"],
        cond=a["conditions"],
        genes=a["genes"],
        mapped=a["gene_mask"],
        adata=ad.read_h5ad(dataset["data"]["h5ad"]),
        train_conditions=train,
        meta=ps.meta,
        training=training_record(ps.meta),
    )


def _row_mean(X, rows: np.ndarray) -> np.ndarray:
    return np.asarray(
        X[rows].mean(0), dtype=np.float64
    ).ravel()  # works for dense and sparse X


def top_de_indices(
    adata: ad.AnnData, conditions, n: int = N_DE
) -> dict[str, np.ndarray]:
    """Top-n DE genes per condition from ``uns[rank_genes_groups_cov_all]`` (GEARS/scGPT convention)."""
    gid = {g: i for i, g in enumerate(adata.var_names)}
    name = adata.obs.drop_duplicates("condition").set_index("condition")[
        "condition_name"
    ]
    ranks = adata.uns["rank_genes_groups_cov_all"]
    return {c: np.array([gid[g] for g in ranks[name[c]][:n]]) for c in conditions}


def compute_means(r: PPResults) -> Means:
    X, oc = r.adata.X, r.adata.obs["condition"].to_numpy()
    conds = r.conditions
    ctrl = _row_mean(X, oc == CTRL)
    train_mean = np.mean([_row_mean(X, oc == c) for c in r.train_conditions], axis=0)
    return Means(
        pred={c: r.pred[r.cond == c].mean(0, dtype=np.float64) for c in conds},
        target={c: r.target[r.cond == c].mean(0, dtype=np.float64) for c in conds},
        h5ad={c: _row_mean(X, oc == c) for c in conds},
        ctrl=ctrl,
        train_mean_delta=train_mean - ctrl,
        de=top_de_indices(r.adata, conds),
    )


# ------------------------------------------------------------------------ metrics


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(pearsonr(a, b)[0])


def recorded_convention_metrics(r: PPResults, m: Means) -> dict[str, float]:
    """Test metrics exactly as recorded by the run: all genes, unmapped genes zeroed."""
    res = {
        k: [] for k in ("pearson", "pearson_de", "pearson_delta", "pearson_de_delta")
    }
    for c in r.conditions:
        p, t, de, ctrl = m.pred[c], m.target[c], m.de[c], m.ctrl
        res["pearson"].append(pearson(p, t))
        res["pearson_de"].append(pearson(p[de], t[de]))
        res["pearson_delta"].append(pearson(p - ctrl, t - ctrl))
        res["pearson_de_delta"].append(pearson(p[de] - ctrl[de], t[de] - ctrl[de]))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["mse"] = float(
        ((r.pred[:, r.mapped] - r.target[:, r.mapped]) ** 2).mean(dtype=np.float64)
    )
    return out


def per_condition_metrics(r: PPResults, m: Means) -> pd.DataFrame:
    """Mapped-gene metrics per test condition, with the train-mean-Δ baseline."""
    mk = r.mapped
    rows = []
    for c in r.conditions:
        s = r.cond == c
        dp, dt = m.pred[c] - m.ctrl, m.target[c] - m.ctrl
        de = m.de[c][mk[m.de[c]]]
        resid = r.pred[s][:, mk] - r.target[s][:, mk]
        rows.append(
            {
                "condition": c,
                "gene": c.split("+")[0],
                "cells": int(s.sum()),
                "pearson_delta_mapped": pearson(dp[mk], dt[mk]),
                "pearson_de_delta_mapped": pearson(dp[de], dt[de]),
                "pearson_delta_trainmean_baseline": pearson(
                    m.train_mean_delta[mk], dt[mk]
                ),
                "pearson_de_delta_trainmean_baseline": pearson(
                    m.train_mean_delta[de], dt[de]
                ),
                "n_de_mapped": len(de),
                "sign_agreement_de": float(np.mean(np.sign(dp[de]) == np.sign(dt[de]))),
                "mse_model": float((resid**2).mean(dtype=np.float64)),
                "mse_control_mean": float(
                    ((m.ctrl[mk] - r.target[s][:, mk]) ** 2).mean(dtype=np.float64)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("condition")


def mean_pairwise_r(profiles: np.ndarray) -> float:
    """Mean off-diagonal Pearson r between rows (how similar the per-condition responses are)."""
    cc = np.corrcoef(profiles)
    return float(cc[~np.eye(len(cc), dtype=bool)].mean())


def verify(r: PPResults, m: Means) -> Checks:
    c = Checks()
    oc = r.adata.obs["condition"].to_numpy()
    mk = r.mapped
    c.add("gene order == h5ad", (r.genes == r.adata.var["gene_name"].to_numpy()).all())
    c.add(
        "cells per condition == h5ad",
        all((r.cond == k).sum() == (oc == k).sum() for k in r.conditions),
    )
    c.add(
        "target means == h5ad means (mapped genes)",
        all(
            np.allclose(m.target[k][mk], m.h5ad[k][mk], atol=1e-5) for k in r.conditions
        ),
    )
    if r.recorded_test:
        # Legacy runs zeroed unmapped genes in predictions and targets and scored all genes.
        c.add(
            "legacy: unmapped genes zeroed in predictions and targets",
            not r.pred[:, ~mk].any() and not r.target[:, ~mk].any(),
        )
        for k, v in recorded_convention_metrics(r, m).items():
            c.close(k, v, r.recorded_test[k], atol=1e-5)
    return c


# ------------------------------------------------------------------------ figures

BLUE, PURPLE, RED, ORANGE, TEAL, GRAY = (
    "#386fa4",
    "#ab6b9d",
    "#d4464c",
    "#e09a62",
    "#3387a4",
    "#8a96a3",
)
METRIC_NOTE = (
    "Δ is the mean expression of a perturbation minus the mean of all control cells (log-normalized). "
    "Only the {n_mapped:,} genes in the DeepSC vocabulary are used."
)


def _fmt_list(items) -> str:
    items = list(items)
    return (
        ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]
        if items
        else "none"
    )


@dataclass
class Example:
    """A representative condition (median mapped-gene Δ correlation) and its derived values."""

    cond: str
    pert: str
    rows: np.ndarray
    pm: np.ndarray
    tm: np.ndarray
    dp: np.ndarray
    dt: np.ndarray
    de: np.ndarray
    r_abs: float
    r_delta: float


def pick_example(r: PPResults, m: Means, pc: pd.DataFrame) -> Example:
    srt = pc.sort_values("pearson_delta_mapped")
    cond = srt.index[len(srt) // 2]
    pm, tm, mi = m.pred[cond], m.target[cond], np.where(r.mapped)[0]
    de = m.de[cond][r.mapped[m.de[cond]]]
    dp, dt = pm - m.ctrl, tm - m.ctrl
    return Example(
        cond,
        cond.split("+")[0],
        r.cond == cond,
        pm,
        tm,
        dp,
        dt,
        de,
        pearson(tm[mi], pm[mi]),
        pearson(dt[mi], dp[mi]),
    )


def fig_absolute(r: PPResults, ex: Example):
    import matplotlib.pyplot as plt

    mi = np.where(r.mapped)[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        ex.tm[mi],
        ex.pm[mi],
        s=5,
        c=BLUE,
        alpha=0.5,
        linewidths=0,
        rasterized=True,
        label="Mapped genes",
    )
    ax.scatter(ex.tm[ex.de], ex.pm[ex.de], s=18, c=RED, label=f"Top-{N_DE} DE genes")
    lim = max(ex.tm[mi].max(), ex.pm[mi].max()) * 1.03
    ax.plot([0, lim], [0, lim], ls="--", c="#888", lw=1)
    ax.set(
        xlabel="Observed mean expression",
        ylabel="Predicted mean expression",
        xlim=(0, lim),
        ylim=(0, lim),
    )
    ax.set_aspect("equal")
    ax.text(
        0.97,
        0.03,
        f"{ex.pert}\nr = {ex.r_abs:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return fig


def fig_delta_scatter(r: PPResults, ex: Example):
    import matplotlib.pyplot as plt

    mi = np.where(r.mapped)[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        ex.dt[mi],
        ex.dp[mi],
        s=5,
        c=PURPLE,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
        label="Mapped genes",
    )
    ax.scatter(ex.dt[ex.de], ex.dp[ex.de], s=18, c=RED, label=f"Top-{N_DE} DE genes")
    for g in ex.de[np.argsort(-np.abs(ex.dt[ex.de]))][:6]:
        ax.annotate(
            r.genes[g],
            (ex.dt[g], ex.dp[g]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    lo, hi = min(ex.dt[mi].min(), ex.dp[mi].min()), max(
        ex.dt[mi].max(), ex.dp[mi].max()
    )
    ax.plot([lo, hi], [lo, hi], ls="--", c="#888", lw=1)
    ax.axhline(0, lw=0.7, c="#aaa")
    ax.axvline(0, lw=0.7, c="#aaa")
    ax.set(xlabel="Observed Δ", ylabel="Predicted Δ")
    ax.text(
        0.97,
        0.03,
        f"{ex.pert}\nr = {ex.r_delta:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return fig


def fig_top_de(r: PPResults, ex: Example):
    import matplotlib.pyplot as plt

    de = ex.de
    pos = np.arange(len(de))
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(de) + 1.2))
    ax.barh(pos - 0.18, ex.dt[de], height=0.36, color=TEAL, label="Observed")
    ax.barh(pos + 0.18, ex.dp[de], height=0.36, color=ORANGE, label="Predicted")
    ax.axvline(0, lw=0.7, c="#666")
    ax.set(yticks=pos, yticklabels=r.genes[de], xlabel=f"Δ under {ex.pert}")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1 * 20 / max(len(de), 1)),
    )
    return fig


def fig_single_cell(r: PPResults, ex: Example, g: int):
    import matplotlib.pyplot as plt
    import seaborn as sns

    ctrl_rows = np.where(r.adata.obs["condition"].to_numpy() == CTRL)[0]
    ctrl_rows = np.random.default_rng(SEED).choice(
        ctrl_rows, size=min(2000, len(ctrl_rows)), replace=False
    )
    Xc = r.adata.X[ctrl_rows][:, [g]]
    ctrl_vals = np.asarray(Xc.todense() if hasattr(Xc, "todense") else Xc).ravel()
    groups = [
        "Control\n(observed)",
        f"{ex.pert}\n(observed)",
        f"{ex.pert}\n(predicted)",
    ]
    vdf = pd.concat(
        [
            pd.DataFrame({"group": groups[0], "expr": ctrl_vals}),
            pd.DataFrame({"group": groups[1], "expr": r.target[ex.rows, g]}),
            pd.DataFrame({"group": groups[2], "expr": r.pred[ex.rows, g]}),
        ]
    )
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.violinplot(
        data=vdf,
        x="group",
        y="expr",
        hue="group",
        palette=["#b8c4ce", TEAL, ORANGE],
        inner="quart",
        cut=0,
        ax=ax,
        legend=False,
    )
    ax.set(xlabel="", ylabel=f"{r.genes[g]} expression (log-normalized)")
    sd = vdf.groupby("group", sort=False).expr.std()
    return fig, {
        "control_cells_sampled": len(ctrl_rows),
        "perturbed_cells": int(ex.rows.sum()),
        "sd": dict(zip(["control", "observed", "predicted"], sd.to_numpy())),
    }


def fig_delta_correlation(pc: pd.DataFrame):
    import matplotlib.pyplot as plt

    o = pc.sort_values("pearson_delta_mapped")
    y = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(o) + 1.5))
    ax.barh(
        y - 0.2,
        o.pearson_delta_mapped,
        height=0.4,
        color=PURPLE,
        label="DeepSC, all mapped genes",
    )
    ax.barh(
        y + 0.2,
        o.pearson_de_delta_mapped,
        height=0.4,
        color=RED,
        label=f"DeepSC, top-{N_DE} DE genes",
    )
    ax.scatter(
        o.pearson_delta_trainmean_baseline,
        y - 0.2,
        marker="|",
        s=120,
        c="k",
        lw=2,
        zorder=3,
        label="Train-mean-Δ baseline",
    )
    ax.scatter(
        o.pearson_de_delta_trainmean_baseline,
        y + 0.2,
        marker="|",
        s=120,
        c="k",
        lw=2,
        zorder=3,
    )
    ax.axvline(0, c="#666", lw=0.7)
    ax.set(
        yticks=y,
        yticklabels=o.gene,
        xlabel="Pearson r between observed and predicted Δ",
        xlim=(-1, 1.02),
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
    )
    return fig


def fig_mse(pc: pd.DataFrame):
    import matplotlib.pyplot as plt

    o = pc.sort_values("mse_model")
    y = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(o) + 1.5))
    ax.hlines(y, o.mse_model, o.mse_control_mean, color="#c9d1d8", lw=2)
    ax.scatter(
        o.mse_control_mean, y, color=GRAY, label="Control-mean baseline", zorder=3
    )
    ax.scatter(o.mse_model, y, color=BLUE, label="DeepSC", zorder=3)
    ax.set(yticks=y, yticklabels=o.gene, xlabel="Per-cell MSE (mapped genes)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
    )
    return fig


def heatmap_genes(r: PPResults, m: Means, order, per_cond: int = 3) -> list[int]:
    return list(
        dict.fromkeys(i for c in order for i in m.de[c][r.mapped[m.de[c]]][:per_cond])
    )


def fig_delta_heatmap(r: PPResults, m: Means, pc: pd.DataFrame, order, sel):
    import matplotlib.pyplot as plt
    import seaborn as sns

    obs = np.array([(m.target[c] - m.ctrl)[sel] for c in order])
    pred = np.array([(m.pred[c] - m.ctrl)[sel] for c in order])
    lim = np.abs(np.r_[obs.ravel(), pred.ravel()]).max()
    ylab = [
        f"{c.split('+')[0]} (r={pc.loc[c, 'pearson_delta_mapped']:.2f})" for c in order
    ]
    fig, ax = plt.subplots(
        2, 1, figsize=(max(10, 0.26 * len(sel) + 3), 11), layout="constrained"
    )
    for a, mat, name in zip(ax, [obs, pred], ["Observed", "Predicted (DeepSC)"]):
        sns.heatmap(
            mat,
            cmap="RdBu_r",
            center=0,
            vmin=-lim,
            vmax=lim,
            xticklabels=r.genes[sel],
            yticklabels=ylab,
            ax=a,
            cbar_kws={"label": "Δ"},
        )
        a.set_ylabel(name, fontsize=11, weight="bold")
        a.tick_params(axis="x", labelsize=7, rotation=90)
        a.tick_params(axis="y", labelsize=8, rotation=0)
    return fig


def fig_history(status: dict, which: str):
    import matplotlib.pyplot as plt

    h, best = status["history"], status["best_epoch"]
    ep = [x["epoch"] for x in h]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    if which == "loss":
        ax.plot(ep, [x["train_mse"] for x in h], "o-", label="Training MSE")
        ax.plot(ep, [x["val"]["mse"] for x in h], "o-", label="Validation MSE")
        ax.plot(
            ep,
            [x["val"]["control_baseline_mse"] for x in h],
            "--",
            c="#888",
            label="Validation, control-input baseline",
        )
        ax.set_ylabel("MSE")
    else:
        ax.plot(ep, [x["val"]["pearson_delta"] for x in h], "o-", label="Pearson Δ")
        ax.plot(
            ep,
            [x["val"]["pearson_de_delta"] for x in h],
            "o-",
            label=f"Pearson Δ (top-{N_DE} DE)",
        )
        ax.set_ylabel("Validation Pearson r")
    ax.axvline(
        best, ls=":", c="#555", lw=1, label=f"Selected checkpoint (epoch {best})"
    )
    ax.set(xlabel="Epoch", xticks=ep)
    ax.legend(frameon=False, fontsize=8)
    return fig


# ----------------------------------------------------------------------- pipeline


def target_knockdown(r: PPResults, m: Means) -> pd.DataFrame:
    """Observed and predicted Δ of each perturbation's own target gene (mapped targets only)."""
    gid = {g: i for i, g in enumerate(r.genes)}
    rows = []
    for c in r.conditions:
        g = gid.get(c.split("+")[0])
        if g is not None and r.mapped[g]:
            rows.append(
                {
                    "condition": c,
                    "observed": m.target[c][g] - m.ctrl[g],
                    "predicted": m.pred[c][g] - m.ctrl[g],
                }
            )
    return (
        pd.DataFrame(rows).set_index("condition")
        if rows
        else pd.DataFrame(columns=["observed", "predicted"])
    )


def run(pred_dir: Path, out: Path, dataset: dict) -> FigureWriter:
    """Verify, then write all PP figures, metadata and tables for one prediction set.

    ``pred_dir`` is a prediction set (see ``deepsc.pipeline.contract``) and
    ``dataset`` its dataset config. Output: ``<out>/pp/<dataset>/<model>/``.
    """
    dataset_label = dataset.get("label", dataset["name"])
    r = load(pred_dir, dataset)
    m = compute_means(r)
    print("Verification:")
    checks = verify(r, m)
    checks.report()
    checks.require()

    pc = per_condition_metrics(r, m)
    ex = pick_example(r, m, pc)
    order = pc.sort_values("pearson_delta_mapped", ascending=False).index
    mk = r.mapped
    sim_obs = mean_pairwise_r(np.array([(m.target[c] - m.ctrl)[mk] for c in order]))
    sim_pred = mean_pairwise_r(np.array([(m.pred[c] - m.ctrl)[mk] for c in order]))
    kd = target_knockdown(r, m)

    apply_style()
    ds, n_cond, n_cells, n_mapped = dataset_label, len(pc), len(r.cond), int(mk.sum())
    w = FigureWriter(
        out,
        "pp",
        dataset["name"],
        ds,
        {
            "predictions": Path(pred_dir).resolve(),
            "h5ad": dataset["data"]["h5ad"],
            "split": dataset["data"]["split"],
            "dataset_config": dataset["_path"],
            "model_config": r.meta["model"].get("config", ""),
        },
        model=r.model_id,
    )
    note = METRIC_NOTE.format(n_mapped=n_mapped)
    setting = (
        f"{ds} test set: {n_cond} perturbations not seen during training ({n_cells:,} cells), "
        f"{len(r.train_conditions)} training perturbations; model {r.model_id}"
    )
    model_r, base_r = (
        pc.pearson_delta_mapped.mean(),
        pc.pearson_delta_trainmean_baseline.mean(),
    )
    model_de, base_de = (
        pc.pearson_de_delta_mapped.mean(),
        pc.pearson_de_delta_trainmean_baseline.mean(),
    )
    better = int((pc.pearson_delta_mapped > pc.pearson_delta_trainmean_baseline).sum())
    better_de = int(
        (pc.pearson_de_delta_mapped > pc.pearson_de_delta_trainmean_baseline).sum()
    )
    negative = list(pc.gene[pc.pearson_delta_mapped < 0])
    rec = r.recorded_test
    summary = {
        "model": r.model_id,
        "split": r.meta["split"],
        "conditions": n_cond,
        "cells": n_cells,
        "mapped_genes": n_mapped,
        "pearson_delta_recorded": rec["pearson_delta"] if rec else None,
        "pearson_de_delta_recorded": rec["pearson_de_delta"] if rec else None,
        "mse_mapped": float(pc.mse_model.mean()),
        "mse_control_mean_baseline": float(pc.mse_control_mean.mean()),
        "pearson_delta_mapped": model_r,
        "pearson_de_delta_mapped": model_de,
        "pearson_delta_trainmean_baseline": base_r,
        "pearson_de_delta_trainmean_baseline": base_de,
        "conditions_better_than_baseline": better,
        "conditions_better_than_baseline_de": better_de,
        "pairwise_r_observed": sim_obs,
        "pairwise_r_predicted": sim_pred,
        "example_condition": ex.cond,
    }
    ex_note = (
        f"{ex.pert} is shown as a representative perturbation: its Δ correlation is the median of the "
        f"{n_cond} test perturbations."
    )

    # Example perturbation -------------------------------------------------------
    w.save(
        fig_absolute(r, ex),
        "example_absolute_expression",
        title=f"Observed vs predicted mean expression, {ex.pert} ({ds})",
        caption_short=f"Mean expression under {ex.pert}: observed vs predicted.",
        caption_long=f"Each dot is one gene: mean expression of the {ex.rows.sum()} {ex.pert} test cells "
        f"(observed) against the mean DeepSC prediction for the same cells; red: top-{N_DE} DE genes. "
        f"Dashed line: y = x. Pearson r = {ex.r_abs:.3f}. {ex_note} Only the {n_mapped:,} genes in "
        f"the DeepSC vocabulary are used.",
        description=f"The high correlation (r = {ex.r_abs:.3f}) mostly reflects that the model reproduces each "
        f"gene's baseline expression level, which a prediction of the unperturbed control would also "
        f"do. It is therefore not evidence that the perturbation effect is captured; the Δ-based "
        f"panels test that.",
        metrics={"condition": ex.cond, "pearson_r": ex.r_abs},
    )
    slope = float(np.polyfit(ex.dt[ex.de], ex.dp[ex.de], 1)[0])
    sign_ok = float(np.mean(np.sign(ex.dp[ex.de]) == np.sign(ex.dt[ex.de])))
    w.save(
        fig_delta_scatter(r, ex),
        "example_delta_scatter",
        title=f"Observed vs predicted perturbation effect, {ex.pert} ({ds})",
        caption_short=f"Expression change under {ex.pert}: observed vs predicted.",
        caption_long=f"Each dot is one gene: observed Δ against predicted Δ for {ex.pert}; red: top-{N_DE} DE "
        f"genes, the six largest labeled. Dashed line: y = x. Pearson r = {ex.r_delta:.3f}. {note}",
        description=f"Removing the control baseline isolates the perturbation effect. For {ex.pert} the "
        f"predicted direction matches the observed one for {sign_ok:.0%} of the top DE genes; the "
        f"least-squares slope of predicted on observed Δ over these genes is {slope:.2f} "
        f"({'effects are underestimated' if slope < 0.9 else 'effect sizes are about right' if slope <= 1.1 else 'effects are overestimated'}).",
        metrics={
            "condition": ex.cond,
            "pearson_r_delta": ex.r_delta,
            "de_sign_agreement": sign_ok,
            "de_slope": slope,
        },
    )
    w.save(
        fig_top_de(r, ex),
        "example_top_de_genes",
        title=f"Top differentially expressed genes under {ex.pert} ({ds})",
        caption_short=f"Observed and predicted Δ of the top DE genes under {ex.pert}.",
        caption_long=f"Observed (teal) and predicted (orange) Δ for the top-{N_DE} DE genes of {ex.pert} "
        f"(ranked by the dataset's differential expression test; {len(ex.de)} of them are in the "
        f"DeepSC vocabulary). {note}",
        description=f"Bars of the same sign indicate a correctly predicted direction ({sign_ok:.0%} of genes). "
        f"Bar length compares effect sizes; the slope of predicted on observed Δ is {slope:.2f}.",
        metrics={
            "condition": ex.cond,
            "genes": list(r.genes[ex.de]),
            "observed": ex.dt[ex.de],
            "predicted": ex.dp[ex.de],
        },
    )
    g = int(ex.de[np.argmax(np.abs(ex.dt[ex.de]))])
    fig, sc_stats = fig_single_cell(r, ex, g)
    sd = sc_stats["sd"]
    w.save(
        fig,
        "example_single_cell_distribution",
        title=f"Single-cell distribution of {r.genes[g]} under {ex.pert} ({ds})",
        caption_short=f"Per-cell {r.genes[g]} expression: control, observed and predicted under {ex.pert}.",
        caption_long=f"Violin plots of log-normalized {r.genes[g]} expression (the top DE gene of {ex.pert} with "
        f"the largest |Δ|) in {sc_stats['control_cells_sampled']:,} randomly sampled control cells, "
        f"the {sc_stats['perturbed_cells']} observed {ex.pert} cells and the DeepSC predictions for "
        f"those cells. Dashed lines: quartiles.",
        description=f"Beyond the mean, the spread of the predictions shows whether cell-to-cell variability is "
        f"captured. The standard deviation of the predicted values is {sd['predicted']:.3f}, "
        f"compared with {sd['observed']:.3f} in the observed perturbed cells"
        + (
            " — the predictions collapse to nearly a single value per gene, i.e. the model predicts "
            "the mean response only."
            if sd["predicted"] < 0.25 * sd["observed"]
            else "."
        ),
        metrics={"condition": ex.cond, "gene": r.genes[g], **sc_stats},
    )

    # All perturbations ----------------------------------------------------------
    w.save(
        fig_delta_correlation(pc),
        "delta_correlation_per_perturbation",
        title=f"Per-perturbation effect correlation vs baseline ({ds})",
        caption_short=f"Δ correlation for each unseen {ds} perturbation, DeepSC vs a train-mean baseline.",
        caption_long=f"For each test perturbation, Pearson r between observed and predicted Δ over all mapped "
        f"genes (purple) and over the top-{N_DE} DE genes (red). Black ticks: baseline that predicts, "
        f"for every test perturbation, the mean Δ of the {len(r.train_conditions)} training "
        f"perturbations. {setting}. {note}",
        description=f"Mean Δ correlation over all mapped genes is {model_r:.3f} for DeepSC and {base_r:.3f} for "
        f"the baseline (DeepSC higher for {better}/{n_cond} perturbations); over the top DE genes "
        f"{model_de:.3f} vs {base_de:.3f} ({better_de}/{n_cond}). "
        + (
            f"Predictions are anti-correlated with the observed response for {_fmt_list(negative)}. "
            if negative
            else ""
        )
        + (
            f"The run's recorded Pearson Δ ({rec['pearson_delta']:.3f}) is higher because it also "
            f"counts {int((~mk).sum())} unmapped genes whose predictions and targets were both set to 0, "
            f"which agree trivially."
            if rec
            else ""
        ),
        metrics=summary,
    )
    worse = list(pc.gene[pc.mse_model >= pc.mse_control_mean])
    w.save(
        fig_mse(pc),
        "mse_per_perturbation",
        title=f"Per-perturbation prediction error vs control-mean baseline ({ds})",
        caption_short=f"Per-cell MSE for each unseen {ds} perturbation, DeepSC vs control mean.",
        caption_long=f"Mean squared error between predicted and observed expression per cell, averaged over the "
        f"cells of each test perturbation (blue), compared with predicting the mean control profile "
        f"for every cell (gray). {setting}. Only the {n_mapped:,} mapped genes are used.",
        description=f"DeepSC has the lower error for {n_cond - len(worse)}/{n_cond} perturbations, mean "
        f"{pc.mse_model.mean():.4f} vs {pc.mse_control_mean.mean():.4f}. The margin is small relative "
        f"to the error itself because most genes change little under any perturbation."
        + (
            f" DeepSC is worse than the control mean for {_fmt_list(worse)}."
            if worse
            else ""
        ),
        metrics={
            "mse_model_mean": pc.mse_model.mean(),
            "mse_control_mean_mean": pc.mse_control_mean.mean(),
            "worse_than_control": worse,
        },
    )
    sel = heatmap_genes(r, m, order)
    kd_ok = int(((kd.observed < 0) & (kd.predicted < 0)).sum()) if len(kd) else 0
    w.save(
        fig_delta_heatmap(r, m, pc, order, sel),
        "delta_heatmap",
        title=f"Observed and predicted responses across perturbations ({ds})",
        caption_short=f"Observed (top) and predicted (bottom) Δ for all unseen {ds} perturbations.",
        caption_long=f"Rows: test perturbations, sorted by Δ correlation (in parentheses); columns: the top-3 "
        f"mapped DE genes of each perturbation ({len(sel)} genes). Both heatmaps share one color "
        f"scale. {note}",
        description=f"Observed responses differ between perturbations (mean pairwise r of Δ profiles over all "
        f"mapped genes {sim_obs:.3f}), whereas predicted responses are "
        f"{'nearly identical' if sim_pred > 0.9 else 'similar' if sim_pred > 0.7 else 'distinct'} "
        f"(mean pairwise r {sim_pred:.3f}). "
        + (
            f"The knock-down of the perturbed gene itself is predicted for {kd_ok}/{len(kd)} "
            f"perturbations whose target is in the vocabulary. "
            if len(kd)
            else ""
        )
        + (
            "Apart from that, the model mostly predicts one shared response, which is why perturbations "
            "with an atypical response score poorly."
            if sim_pred > 0.9
            else ""
        ),
        metrics={
            "pairwise_r_observed": sim_obs,
            "pairwise_r_predicted": sim_pred,
            "genes": list(r.genes[sel]),
            "target_knockdown": kd.round(4).to_dict(orient="index"),
        },
    )

    if r.training:
        _save_history(w, r.training, ds)
    w.table(pc.round(5), "per_condition_metrics")
    if len(kd):
        w.table(kd.round(5), "target_knockdown")
    w.finalize(checks, summary=summary)
    return w


def _save_history(w: FigureWriter, training: dict, ds: str) -> None:
    h, best = training["history"], training["best_epoch"]
    bv = next(x["val"] for x in h if x["epoch"] == best)
    mapped = training.get("metric_convention") == "mapped_genes"
    genes_txt = (
        "over the genes the model represents"
        if mapped
        else "as logged by the run (unmapped genes included)"
    )
    w.save(
        fig_history(training, "loss"),
        "training_loss",
        title=f"PP fine-tuning loss ({ds})",
        caption_short=f"Training and validation MSE during PP fine-tuning on {ds}.",
        caption_long=f"Per-epoch MSE on the training and validation sets; dashed: validation error of the "
        f"control-input baseline as logged by the run. Dotted line: selected checkpoint (epoch {best}, "
        f"highest validation Pearson Δ). Training stopped after epoch {h[-1]['epoch']}.",
        description=f"Validation MSE stays within {min(x['val']['mse'] for x in h):.4f}–"
        f"{max(x['val']['mse'] for x in h):.4f} over all epochs while training MSE falls from "
        f"{h[0]['train_mse']:.4f} to {h[-1]['train_mse']:.4f}.",
        metrics={"history": h, "best_epoch": best},
    )
    w.save(
        fig_history(training, "val"),
        "training_validation_delta",
        title=f"PP validation Δ correlation during fine-tuning ({ds})",
        caption_short=f"Validation Pearson Δ during PP fine-tuning on {ds}.",
        caption_long=f"Pearson Δ over all genes and over the top-{N_DE} DE genes on the validation perturbations "
        f"after each epoch, {genes_txt}. Dotted line: selected checkpoint (epoch {best}).",
        description=f"The selected epoch has validation Pearson Δ {bv['pearson_delta']:.3f} (DE "
        f"{bv['pearson_de_delta']:.3f})."
        + (
            ""
            if mapped
            else " The logged values include unmapped genes and are therefore higher than the mapped-gene "
            "test values reported in the other figures."
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
        help="root dir; writes <out>/pp/<dataset>/<model>/",
    )
    args = p.parse_args(argv)
    use_headless_backend()
    dataset = (
        args.dataset
        or json.loads((args.predictions / "meta.json").read_text())["dataset"]
    )
    w = run(args.predictions, args.out, load_dataset_config("pp", dataset))
    s = json.loads((w.root / "manifest.json").read_text())["summary"]
    print(
        f"Pearson Δ (mapped) {s['pearson_delta_mapped']:.4f} | DE {s['pearson_de_delta_mapped']:.4f} | "
        f"train-mean-Δ baseline {s['pearson_delta_trainmean_baseline']:.4f} | MSE {s['mse_mapped']:.4f}"
    )
    print(f"Saved {len(w.records)} figures to {w.root.resolve()}")


if __name__ == "__main__":
    main()
