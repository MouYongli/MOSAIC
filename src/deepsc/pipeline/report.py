"""Turn prediction sets into figures, tables and captions, per model and compared.

    deepsc report --task cta --dataset zheng                 # every model found
    deepsc report --task pp  --dataset adamson --models a b  # selected models

Reads ``<infer-root>/<task>/<dataset>/<model>/<split>/`` (default
``outputs/infer``) and writes ``<out>/<task>/<dataset>/`` (default
``outputs/viz``):

    <model>/       figures, tables, README.md, manifest.json for each model
    comparison/    metrics of all models side by side (table + figures)
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import argparse
from deepsc.pipeline.infer import DEFAULT_OUT as DEFAULT_INFER_ROOT
from deepsc.pipeline.registry import load_dataset_config

DEFAULT_OUT = Path("outputs") / "viz"


def discover(
    task: str, dataset: str, infer_root: Path, split: str, models: list[str] | None
) -> list[Path]:
    root = Path(infer_root) / task / dataset
    found = sorted(p.parent for p in root.glob(f"*/{split}/meta.json"))
    if models:
        by_id = {p.parent.name: p for p in found}
        missing = [m for m in models if m not in by_id]
        if missing:
            raise FileNotFoundError(
                f"no {split} predictions for {missing} under {root}; found {sorted(by_id)}"
            )
        found = [by_id[m] for m in models]
    if not found:
        raise FileNotFoundError(
            f"no prediction sets under {root}/*/{split}; run `deepsc infer` first"
        )
    return found


# ------------------------------------------------------------------ comparison


def _bar_groups(
    ax,
    table: pd.DataFrame,
    cols: list[str],
    labels: list[str],
    baseline: dict | None = None,
):
    """Grouped bars: one group per metric, one bar per model; optional baseline ticks."""
    import seaborn as sns

    n = len(table)
    width = 0.8 / max(n, 1)
    colors = sns.color_palette("tab10", n)
    x = np.arange(len(cols))
    for i, (model, row) in enumerate(table.iterrows()):
        xs = x - 0.4 + width * (i + 0.5)
        ax.bar(
            xs,
            row[cols].to_numpy(float),
            width=width * 0.95,
            color=colors[i],
            label=model,
        )
        for xi, v in zip(xs, row[cols]):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    labelled = set()
    for j, c in enumerate(cols):
        if baseline and c in baseline:
            lab = baseline[c]["label"]
            ax.hlines(
                baseline[c]["value"],
                x[j] - 0.42,
                x[j] + 0.42,
                colors="k",
                linestyles="--",
                lw=1.2,
                label=None if lab in labelled else lab,
            )
            labelled.add(lab)
    ax.set_xticks(x, labels)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
    )
    ax.grid(axis="x", visible=False)


def compare_cta(writers, dataset_cfg, out) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    from deepsc.viz._common import Checks, FigureWriter, apply_style

    summaries = {
        w.model: json.loads((w.root / "manifest.json").read_text())["summary"]
        for w in writers
    }
    table = pd.DataFrame(
        {
            m: {
                "cells": s["cells"],
                "accuracy": s["accuracy"],
                "macro_f1": s["macro_f1"],
                "weighted_f1": s["weighted_f1"],
            }
            for m, s in summaries.items()
        }
    ).T
    per_class = pd.DataFrame({m: s["per_class_f1"] for m, s in summaries.items()}).T
    ds = dataset_cfg.get("label", dataset_cfg["name"])
    apply_style()
    w = FigureWriter(
        out,
        "cta",
        dataset_cfg["name"],
        ds,
        {"predictions": ", ".join(summaries)},
        model="comparison",
    )
    models_txt = ", ".join(
        f"{m} (accuracy {r.accuracy:.1%}, macro-F1 {r.macro_f1:.3f})"
        for m, r in table.iterrows()
    )
    best = table.macro_f1.idxmax()

    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(table) * 3 / 2, 4))
    _bar_groups(
        ax,
        table,
        ["accuracy", "macro_f1", "weighted_f1"],
        ["Accuracy", "Macro-F1", "Weighted F1"],
    )
    ax.set(ylabel="Score", ylim=(0, 1.08))
    w.save(
        fig,
        "metrics",
        title=f"CTA performance by model, {ds}",
        caption_short=f"Cell type annotation performance of {len(table)} model(s) on the {ds} test set.",
        caption_long=f"Accuracy, macro-F1 (unweighted mean of per-class F1) and weighted F1 (weighted by class "
        f"size) on the {ds} test set ({int(table.cells.iloc[0]):,} cells).",
        description=f"{models_txt}. Highest macro-F1: {best}.",
        metrics=table.round(4).to_dict(orient="index"),
    )

    fig, ax = plt.subplots(
        figsize=(0.7 * per_class.shape[1] + 3, 0.5 * len(per_class) + 1.8)
    )
    sns.heatmap(
        per_class,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "F1"},
        ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.set(xlabel="Cell type", ylabel="Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    spread = (per_class.max() - per_class.min()).sort_values(ascending=False)
    w.save(
        fig,
        "per_class_f1",
        title=f"Per-class F1 by model, {ds}",
        caption_short=f"Per-class F1 of each model on the {ds} test set.",
        caption_long="Rows: models; columns: reference cell types; entries: F1 of that class.",
        description=(
            f"Classes where the models differ most: "
            f"{', '.join(f'{c} (range {v:.2f})' for c, v in spread.head(3).items())}."
            if len(per_class) > 1
            else f"Single model; lowest F1 for {', '.join(per_class.iloc[0].nsmallest(3).index)}."
        ),
        metrics=per_class.round(4).to_dict(orient="index"),
    )
    w.table(table.round(4), "metrics")
    w.table(per_class.round(4), "per_class_f1")
    w.finalize(
        Checks({"all per-model reports verified": True}),
        summary={"models": list(table.index)},
    )
    return w.root


def compare_pp(writers, dataset_cfg, out) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    from deepsc.viz._common import Checks, FigureWriter, apply_style

    summaries = {
        w.model: json.loads((w.root / "manifest.json").read_text())["summary"]
        for w in writers
    }
    cols = ["pearson_delta_mapped", "pearson_de_delta_mapped", "mse_mapped"]
    table = pd.DataFrame({m: {c: s[c] for c in cols} for m, s in summaries.items()}).T
    first = next(iter(summaries.values()))
    base = {
        "pearson_delta_mapped": first["pearson_delta_trainmean_baseline"],
        "pearson_de_delta_mapped": first["pearson_de_delta_trainmean_baseline"],
        "mse_mapped": first["mse_control_mean_baseline"],
    }
    per_cond = pd.DataFrame(
        {
            w.model: pd.read_csv(
                w.table_dir / f"pp_{w.dataset}_{w.model}_per_condition_metrics.csv",
                index_col=0,
            )["pearson_delta_mapped"]
            for w in writers
        }
    ).T
    ds = dataset_cfg.get("label", dataset_cfg["name"])
    apply_style()
    w = FigureWriter(
        out,
        "pp",
        dataset_cfg["name"],
        ds,
        {"predictions": ", ".join(summaries)},
        model="comparison",
    )
    n_cond = per_cond.shape[1]

    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(table) * 2, 4))
    _bar_groups(
        ax,
        table,
        cols[:2],
        ["Pearson Δ", "Pearson Δ (top-20 DE)"],
        {c: {"value": base[c], "label": "Train-mean-Δ baseline"} for c in cols[:2]},
    )
    ax.set(
        ylabel="Mean Pearson r over perturbations",
        ylim=(min(0, table[cols[:2]].min().min()) - 0.05, 1.08),
    )
    beats = {m: int((per_cond.loc[m] > 0).sum()) for m in table.index}
    w.save(
        fig,
        "delta_correlation",
        title=f"PP effect correlation by model, {ds}",
        caption_short=f"Perturbation-effect correlation of each model on {ds}, with a train-mean baseline.",
        caption_long=f"Mean over the {n_cond} test perturbations of the Pearson r between observed and predicted "
        f"Δ (perturbation minus control mean), over all genes the models represent and over the "
        f"top-20 DE genes. Dashed: baseline predicting the mean Δ of the training perturbations.",
        description="; ".join(
            f"{m}: {r.pearson_delta_mapped:.3f} / DE {r.pearson_de_delta_mapped:.3f} "
            f"({'above' if r.pearson_delta_mapped > base['pearson_delta_mapped'] else 'below'} the baseline "
            f"{base['pearson_delta_mapped']:.3f})"
            for m, r in table.iterrows()
        )
        + ".",
        metrics={
            "models": table.round(4).to_dict(orient="index"),
            "baseline": base,
            "positive_conditions": beats,
        },
    )

    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(table), 4))
    _bar_groups(
        ax,
        table,
        cols[2:],
        ["MSE"],
        {"mse_mapped": {"value": base["mse_mapped"], "label": "Control-mean baseline"}},
    )
    ax.set(ylabel="Per-cell MSE (mapped genes)")
    w.save(
        fig,
        "mse",
        title=f"PP prediction error by model, {ds}",
        caption_short=f"Per-cell MSE of each model on {ds}, with a control-mean baseline.",
        caption_long=f"Mean squared error between predicted and observed expression, averaged over cells of "
        f"each test perturbation and then over the {n_cond} perturbations. Dashed: predicting the "
        f"mean control profile.",
        description="; ".join(f"{m}: {r.mse_mapped:.4f}" for m, r in table.iterrows())
        + f"; control-mean baseline {base['mse_mapped']:.4f}.",
        metrics={"models": table.round(5).to_dict(orient="index"), "baseline": base},
    )

    order = per_cond.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(0.45 * n_cond + 3, 0.5 * len(per_cond) + 2))
    sns.heatmap(
        per_cond[order],
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r of Δ"},
        ax=ax,
        annot_kws={"fontsize": 6},
        xticklabels=[c.split("+")[0] for c in order],
    )
    ax.set(xlabel="Test perturbation", ylabel="Model")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    neg = [c.split("+")[0] for c in order if (per_cond[c] < 0).all()]
    w.save(
        fig,
        "delta_correlation_per_perturbation",
        title=f"Per-perturbation effect correlation by model, {ds}",
        caption_short=f"Δ correlation per test perturbation and model on {ds}.",
        caption_long="Rows: models; columns: test perturbations sorted by mean correlation; entries: Pearson r "
        "between observed and predicted Δ over mapped genes.",
        description=(
            f"Perturbations predicted in the wrong direction by every model: {', '.join(neg)}."
            if neg
            else "No perturbation is anti-correlated for every model."
        ),
        metrics=per_cond.round(4).to_dict(orient="index"),
    )
    w.table(table.round(5), "metrics")
    w.table(per_cond.round(4), "delta_correlation_per_perturbation")
    w.finalize(
        Checks({"all per-model reports verified": True}),
        summary={"models": list(table.index), "baselines": base},
    )
    return w.root


# -------------------------------------------------------------------------- run


def run(
    task: str,
    dataset: str,
    models=None,
    split="test",
    infer_root=DEFAULT_INFER_ROOT,
    out=DEFAULT_OUT,
    comparison=True,
) -> list[Path]:
    from deepsc.viz import cta as viz_cta
    from deepsc.viz import perturbation as viz_pp
    from deepsc.viz._common import use_headless_backend

    use_headless_backend()
    cfg = load_dataset_config(task, dataset)
    viz = {"cta": viz_cta, "pp": viz_pp}[task]
    writers = []
    for pred_dir in discover(task, cfg["name"], infer_root, split, models):
        print(f"== {pred_dir}")
        writers.append(viz.run(pred_dir, out, cfg))  # raises if verification fails
    roots = [w.root for w in writers]
    if comparison:
        roots.append({"cta": compare_cta, "pp": compare_pp}[task](writers, cfg, out))
    return roots


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, choices=["cta", "pp"])
    p.add_argument("--dataset", required=True, help="dataset config name or YAML path")
    p.add_argument(
        "--models",
        nargs="+",
        help="model ids to include (default: all with predictions)",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--infer-root", type=Path, default=DEFAULT_INFER_ROOT)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="writes <out>/<task>/<dataset>/{<model>,comparison}",
    )
    p.add_argument("--no-comparison", dest="comparison", action="store_false")


def main_from_args(args) -> None:
    for root in run(
        args.task,
        args.dataset,
        args.models,
        args.split,
        args.infer_root,
        args.out,
        args.comparison,
    ):
        print(f"Wrote {root}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="deepsc report", description=__doc__.split("\n\n")[0]
    )
    add_arguments(p)
    main_from_args(p.parse_args(argv))


if __name__ == "__main__":
    main()
