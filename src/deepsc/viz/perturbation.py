"""Perturbation prediction (PP) figures from saved DeepSC test predictions.

Usage:
    python -m deepsc.viz.perturbation [--run-dir DIR] [--h5ad FILE] [--split PKL] [--out DIR]

Inputs:
  * ``<run-dir>/test_predictions.npz``: per test cell ``predictions`` and ``targets``
    (cells x genes), ``conditions``, ``genes`` and ``gene_vocab_ids`` (0 = unmapped);
  * ``<run-dir>/status.json``: run record with test metrics and epoch history;
  * the GEARS ``perturb_processed.h5ad`` (control cells, ``uns[rank_genes_groups_cov_all]``);
  * the GEARS split pickle (training perturbations, for the train-mean baseline).

Outputs (in ``--out``): ``pp_prediction_comparison.png``, ``pp_delta_heatmap.png``,
``pp_training_history.png`` and ``pp_per_condition_metrics.csv``.

Genes without a DeepSC vocabulary entry have predictions AND targets set to 0 in the
saved arrays. With the original metric their delta is -control on both sides, which
inflates delta correlations. Verification reproduces the recorded metrics with that
convention; every figure and the "mapped" metrics use mapped genes only.

Baseline "train-mean Δ": predict the average Δ of all training perturbations for
every test perturbation (no learning involved).
"""

from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import argparse
from deepsc.viz._common import (
    DOWNSTREAM_RUN_ROOT,
    Checks,
    read_status,
    savefig,
    use_headless_backend,
)

DEFAULT_RUN_DIR = DOWNSTREAM_RUN_ROOT / "pp_adamson_1215"
DEFAULT_DATA_DIR = Path("/DATA2/DeepSC/adamson")
DEFAULT_H5AD = DEFAULT_DATA_DIR / "perturb_processed.h5ad"
DEFAULT_SPLIT = DEFAULT_DATA_DIR / "splits/adamson_simulation_1_0.75.pkl"
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
    mapped: np.ndarray  # bool per gene: has a DeepSC vocabulary id
    status: dict
    adata: ad.AnnData
    train_conditions: list[str]

    @property
    def conditions(self) -> np.ndarray:
        return np.unique(self.cond)


@dataclass
class Means:
    """Per-condition mean profiles, computed once and shared by checks and plots."""

    pred: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    h5ad: dict[str, np.ndarray]  # same test conditions, straight from the h5ad
    ctrl: np.ndarray
    train_mean_delta: np.ndarray
    de: dict[str, np.ndarray]  # top-N DE gene indices per test condition


def load(
    run_dir: Path = DEFAULT_RUN_DIR,
    h5ad: Path = DEFAULT_H5AD,
    split: Path = DEFAULT_SPLIT,
) -> PPResults:
    z = np.load(Path(run_dir) / "test_predictions.npz")
    with open(split, "rb") as f:
        train = [c for c in pickle.load(f)["train"] if c != CTRL]
    return PPResults(
        pred=z["predictions"],
        target=z["targets"],
        cond=z["conditions"],
        genes=z["genes"],
        mapped=z["gene_vocab_ids"] > 0,
        status=read_status(run_dir),
        adata=ad.read_h5ad(h5ad),
        train_conditions=train,
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
        f"cells {len(r.cond)} == recorded {r.status['counts']['test']}",
        len(r.cond) == r.status["counts"]["test"],
    )
    c.add(
        f"mapped genes {mk.sum()} == recorded {r.status['mapped_genes']}",
        mk.sum() == r.status["mapped_genes"],
    )
    c.add(
        "unmapped genes: predictions and targets all 0",
        not r.pred[:, ~mk].any() and not r.target[:, ~mk].any(),
    )
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
    for k, v in recorded_convention_metrics(r, m).items():
        c.close(k, v, r.status["test"][k], atol=1e-5)
    return c


# -------------------------------------------------------------------------- plots


def plot_overview(r: PPResults, m: Means, pc: pd.DataFrame, path: Path) -> dict:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.95)
    mk, mi, genes = r.mapped, np.where(r.mapped)[0], r.genes
    # Representative example: the condition with the median mapped-gene Δ correlation.
    srt = pc.sort_values("pearson_delta_mapped")
    cond = srt.index[len(srt) // 2]
    s = r.cond == cond
    pm, tm = m.pred[cond], m.target[cond]
    dp, dt = pm - m.ctrl, tm - m.ctrl
    de = m.de[cond][mk[m.de[cond]]]
    r_abs, r_d = pearson(tm[mi], pm[mi]), pearson(dt[mi], dp[mi])
    pert = cond.split("+")[0]

    fig, ax = plt.subplots(2, 3, figsize=(19, 11.5), layout="constrained")
    fig.suptitle(
        f"Perturbation prediction · DeepSC · {len(pc)} unseen test perturbations "
        f"({len(r.cond):,} cells, {mk.sum():,} mapped genes)",
        fontsize=16,
        weight="bold",
    )

    a = ax[0, 0]  # absolute means: dominated by baseline expression
    a.scatter(
        tm[mi], pm[mi], s=6, c="#386fa4", alpha=0.5, linewidths=0, rasterized=True
    )
    a.scatter(tm[de], pm[de], s=22, c="#d4464c", label=f"Top-{N_DE} DE genes (mapped)")
    lim = max(tm[mi].max(), pm[mi].max()) * 1.03
    a.plot([0, lim], [0, lim], ls="--", c="#888", lw=1)
    a.set(
        xlabel="Observed mean expression",
        ylabel="Predicted mean expression",
        title=f"A · Absolute expression · {cond} · r={r_abs:.3f}",
        xlim=(0, lim),
        ylim=(0, lim),
    )
    a.legend(frameon=False, loc="upper left")

    b = ax[0, 1]  # perturbation effect
    b.scatter(
        dt[mi], dp[mi], s=6, c="#ab6b9d", alpha=0.45, linewidths=0, rasterized=True
    )
    b.scatter(dt[de], dp[de], s=22, c="#d4464c")
    for g in de[np.argsort(-np.abs(dt[de]))][:6]:
        b.annotate(
            genes[g],
            (dt[g], dp[g]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    lo, hi = min(dt[mi].min(), dp[mi].min()), max(dt[mi].max(), dp[mi].max())
    b.plot([lo, hi], [lo, hi], ls="--", c="#888", lw=1)
    b.axhline(0, lw=0.7, c="#aaa")
    b.axvline(0, lw=0.7, c="#aaa")
    b.set(
        xlabel="Observed Δ vs control mean",
        ylabel="Predicted Δ vs control mean",
        title=f"B · Perturbation effect · {cond} · r={r_d:.3f}",
    )

    c_ = ax[0, 2]  # top DE genes
    pos = np.arange(len(de))
    c_.barh(pos - 0.18, dt[de], height=0.36, color="#3387a4", label="Observed Δ")
    c_.barh(pos + 0.18, dp[de], height=0.36, color="#e09a62", label="Predicted Δ")
    c_.axvline(0, lw=0.7, c="#666")
    lo, hi = min(dt[de].min(), dp[de].min(), 0), max(dt[de].max(), dp[de].max(), 0)
    c_.set(
        yticks=pos,
        yticklabels=genes[de],
        xlim=(lo * 1.05, hi + 0.45 * (hi - lo)),
        xlabel="Mean expression difference vs control",
        title=f"C · Top-{N_DE} DE genes (mapped: {len(de)}) · {cond}",
    )
    c_.invert_yaxis()
    c_.legend(frameon=False, loc="lower right")
    c_.tick_params(axis="y", labelsize=8)

    d = ax[1, 0]  # single-cell distributions of the strongest DE gene
    g = de[np.argmax(np.abs(dt[de]))]
    ctrl_rows = np.where(r.adata.obs["condition"].to_numpy() == CTRL)[0]
    ctrl_rows = np.random.default_rng(SEED).choice(
        ctrl_rows, size=min(2000, len(ctrl_rows)), replace=False
    )
    ctrl_vals = np.asarray(
        r.adata.X[ctrl_rows][:, [g]].todense()
        if hasattr(r.adata.X, "todense")
        else r.adata.X[ctrl_rows][:, [g]]
    ).ravel()
    vdf = pd.concat(
        [
            pd.DataFrame({"group": "Control (observed)", "expr": ctrl_vals}),
            pd.DataFrame({"group": f"{pert} (observed)", "expr": r.target[s, g]}),
            pd.DataFrame({"group": f"{pert} (predicted)", "expr": r.pred[s, g]}),
        ]
    )
    sns.violinplot(
        data=vdf,
        x="group",
        y="expr",
        hue="group",
        palette=["#b8c4ce", "#3387a4", "#e09a62"],
        inner="quart",
        cut=0,
        ax=d,
        legend=False,
    )
    d.set(
        xlabel=f"Cells (control: random {len(ctrl_rows):,}; perturbed: {s.sum():,})",
        ylabel=f"{genes[g]} expression (log-normalized)",
        title=f"D · Single-cell distribution · {genes[g]} under {cond}",
    )

    e = ax[1, 1]  # per-condition Δ correlation vs baseline
    o = pc.sort_values("pearson_delta_mapped")
    y = np.arange(len(o))
    e.barh(
        y - 0.2,
        o.pearson_delta_mapped,
        height=0.4,
        color="#ab6b9d",
        label="All mapped genes",
    )
    e.barh(
        y + 0.2,
        o.pearson_de_delta_mapped,
        height=0.4,
        color="#d4464c",
        label=f"Top-{N_DE} DE (mapped)",
    )
    e.scatter(
        o.pearson_delta_trainmean_baseline,
        y - 0.2,
        marker="|",
        s=140,
        c="k",
        lw=2,
        zorder=3,
        label="Train-mean-Δ baseline",
    )
    e.scatter(
        o.pearson_de_delta_trainmean_baseline,
        y + 0.2,
        marker="|",
        s=140,
        c="k",
        lw=2,
        zorder=3,
    )
    e.axvline(o.pearson_delta_mapped.mean(), ls="--", c="#ab6b9d", lw=1)
    e.axvline(o.pearson_de_delta_mapped.mean(), ls="--", c="#d4464c", lw=1)
    e.axvline(0, c="#666", lw=0.7)
    e.set(
        yticks=y,
        yticklabels=o.gene,
        xlabel="Pearson r of Δ (observed vs predicted)",
        title=f"E · Δ correlation per perturbation · DeepSC {o.pearson_delta_mapped.mean():.3f} "
        f"vs baseline {o.pearson_delta_trainmean_baseline.mean():.3f}",
    )
    e.legend(frameon=False, loc="upper left", fontsize=8)
    e.tick_params(axis="y", labelsize=8)

    f = ax[1, 2]  # MSE vs control-mean baseline
    o2 = pc.sort_values("mse_model")
    y2 = np.arange(len(o2))
    f.hlines(y2, o2.mse_model, o2.mse_control_mean, color="#c9d1d8", lw=2)
    f.scatter(
        o2.mse_control_mean,
        y2,
        color="#8a96a3",
        label="Control-mean baseline",
        zorder=3,
    )
    f.scatter(o2.mse_model, y2, color="#386fa4", label="DeepSC", zorder=3)
    f.set(
        yticks=y2,
        yticklabels=o2.gene,
        xlabel="Per-cell MSE over mapped genes",
        title=f"F · MSE per perturbation · DeepSC lower in {(o2.mse_model < o2.mse_control_mean).sum()}/{len(o2)}",
    )
    f.legend(frameon=False, loc="lower right", fontsize=8)
    f.tick_params(axis="y", labelsize=8)

    savefig(fig, path, dpi=165)
    return {"condition": cond, "r_abs": r_abs, "r_delta": r_d, "violin_gene": genes[g]}


def plot_heatmap(
    r: PPResults, m: Means, pc: pd.DataFrame, path: Path, per_cond: int = 3
) -> dict:
    import matplotlib.pyplot as plt
    import seaborn as sns

    mk = r.mapped
    order = pc.sort_values("pearson_delta_mapped", ascending=False).index
    sel = list(dict.fromkeys(i for c in order for i in m.de[c][mk[m.de[c]]][:per_cond]))
    obs = np.array([(m.target[c] - m.ctrl)[sel] for c in order])
    pred = np.array([(m.pred[c] - m.ctrl)[sel] for c in order])
    sim_obs = mean_pairwise_r(np.array([(m.target[c] - m.ctrl)[mk] for c in order]))
    sim_pred = mean_pairwise_r(np.array([(m.pred[c] - m.ctrl)[mk] for c in order]))
    lim = np.abs(np.r_[obs.ravel(), pred.ravel()]).max()
    ylab = [
        f"{c.split('+')[0]} (r={pc.loc[c, 'pearson_delta_mapped']:.2f})" for c in order
    ]
    fig, ax = plt.subplots(
        2, 1, figsize=(max(14, 0.3 * len(sel) + 4), 13), layout="constrained"
    )
    for a, mat, title in zip(
        ax, [obs, pred], ["A · Observed Δ", "B · Predicted Δ (DeepSC)"]
    ):
        sns.heatmap(
            mat,
            cmap="RdBu_r",
            center=0,
            vmin=-lim,
            vmax=lim,
            xticklabels=r.genes[sel],
            yticklabels=ylab,
            ax=a,
            cbar_kws={"label": "Δ mean expression vs control"},
        )
        a.set_title(title, loc="left", fontsize=13)
        a.tick_params(axis="x", labelsize=7, rotation=90)
        a.tick_params(axis="y", labelsize=8, rotation=0)
    fig.suptitle(
        f"Test perturbations × top-{per_cond} DE genes each (mapped, {len(sel)} genes) · rows sorted by Δ "
        f"correlation\nMean pairwise r between perturbations (all mapped genes): observed {sim_obs:.3f} · "
        f"predicted {sim_pred:.3f}",
        fontsize=14,
        weight="bold",
    )
    savefig(fig, path, dpi=150)
    return {"pairwise_r_observed": sim_obs, "pairwise_r_predicted": sim_pred}


def plot_history(status: dict, title: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    h, best = status["history"], status["best_epoch"]
    ep = [x["epoch"] for x in h]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    ax[0].plot(ep, [x["train_mse"] for x in h], "o-", label="Train MSE")
    ax[0].plot(ep, [x["val"]["mse"] for x in h], "o-", label="Validation MSE")
    ax[0].plot(
        ep,
        [x["val"]["control_baseline_mse"] for x in h],
        "--",
        c="#888",
        label="Validation control-input baseline",
    )
    for k, lab in [
        ("pearson_delta", "Pearson Δ"),
        ("pearson_de_delta", "Pearson Δ (DE)"),
    ]:
        ax[1].plot(ep, [x["val"][k] for x in h], "o-", label=lab)
    for a in ax:
        a.axvline(best, ls="--", c="#888", lw=1)
        a.set(xlabel="Epoch", xticks=ep)
        a.grid(alpha=0.25)
    ax[0].legend(frameon=False, fontsize=9, loc="center right")
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")
    ax[0].set(ylabel="MSE", title="A · Loss")
    ax[1].set(ylabel="Pearson r")
    ax[1].set_title(
        f"B · Validation Δ correlation (best = epoch {best}; as logged, incl. unmapped genes)",
        fontsize=10,
    )
    fig.suptitle(title, fontsize=14, weight="bold")
    return savefig(fig, path, dpi=160)


# ---------------------------------------------------------------------------- CLI


def run(
    run_dir: Path, h5ad: Path, split: Path, out: Path, dataset_label: str = "Adamson"
) -> dict:
    """Verify, then write all PP figures and tables to ``out``. Returns a summary dict."""
    out.mkdir(parents=True, exist_ok=True)
    r = load(run_dir, h5ad, split)
    m = compute_means(r)
    print("Verification:")
    checks = verify(r, m)
    checks.report()
    checks.require()

    pc = per_condition_metrics(r, m)
    pc.round(5).to_csv(out / "pp_per_condition_metrics.csv")
    example = plot_overview(r, m, pc, out / "pp_prediction_comparison.png")
    similarity = plot_heatmap(r, m, pc, out / "pp_delta_heatmap.png")
    plot_history(
        r.status,
        f"PP fine-tuning history · {dataset_label}",
        out / "pp_training_history.png",
    )
    return {
        "recorded": r.status["test"],
        "per_condition": pc,
        "n_train": len(r.train_conditions),
        "example": example,
        **similarity,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="dir with test_predictions.npz, status.json",
    )
    p.add_argument(
        "--h5ad", type=Path, default=DEFAULT_H5AD, help="GEARS perturb_processed.h5ad"
    )
    p.add_argument(
        "--split", type=Path, default=DEFAULT_SPLIT, help="GEARS split pickle"
    )
    p.add_argument("--out", type=Path, default=Path("viz_outputs/pp"))
    p.add_argument("--dataset-label", default="Adamson")
    args = p.parse_args(argv)
    use_headless_backend()
    s = run(args.run_dir, args.h5ad, args.split, args.out, args.dataset_label)

    pc, t = s["per_condition"], s["recorded"]
    better = (pc.pearson_delta_mapped > pc.pearson_delta_trainmean_baseline).sum()
    print(f"\n{'metric':<24}{'recorded':>10}{'mapped':>10}{'baseline':>10}")
    print(
        f"{'Pearson Δ':<24}{t['pearson_delta']:>10.4f}{pc.pearson_delta_mapped.mean():>10.4f}"
        f"{pc.pearson_delta_trainmean_baseline.mean():>10.4f}"
    )
    print(
        f"{'Pearson Δ (top-20 DE)':<24}{t['pearson_de_delta']:>10.4f}{pc.pearson_de_delta_mapped.mean():>10.4f}"
        f"{pc.pearson_de_delta_trainmean_baseline.mean():>10.4f}"
    )
    print(
        f"Baseline = mean Δ of {s['n_train']} training perturbations; DeepSC better in {better}/{len(pc)}."
    )
    print(
        f"Mean pairwise r of Δ between perturbations: observed {s['pairwise_r_observed']:.3f}, "
        f"predicted {s['pairwise_r_predicted']:.3f}"
    )
    ex = s["example"]
    print(
        f"Example (median Δ r): {ex['condition']} | absolute r={ex['r_abs']:.3f}, Δ r={ex['r_delta']:.3f}"
    )
    print("Saved to", args.out.resolve())


if __name__ == "__main__":
    main()
