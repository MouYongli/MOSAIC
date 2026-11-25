"""Fine-tune a pre-trained model on a benchmark dataset.

    deepsc finetune --task cta --dataset zheng   --model deepsc_pretrain_1215
    deepsc finetune --task pp  --dataset adamson --model deepsc_pretrain_1215

Writes ``<out>/<task>/<dataset>/<run_id>/`` (default ``outputs/finetune``):

    best_model.pt   checkpoint with the best validation score ({"model", "epoch", "val", ...})
    status.json     run record: config, per-epoch history, best epoch (read by ``report``)
    model.yaml      model config of the result; pass it to ``deepsc infer --model``
    train.log       progress log (one JSON object per line)

Selection: CTA keeps the epoch with the lowest validation cross-entropy, PP the
epoch with the highest validation Pearson delta (mapped genes). Training stops
early after ``--patience`` epochs without improvement.

CTA reuses ``deepsc.finetune.cell_type_annotation.CellTypeAnnotation`` (data,
model loading, optimizer, training step); PP follows the training step of
``deepsc/finetune/finetune_perturbation.py``.
"""

from __future__ import annotations
import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np

import argparse
import time
from datetime import datetime, timezone
from deepsc.pipeline import models
from deepsc.pipeline.contract import code_version
from deepsc.pipeline.registry import load_dataset_config, load_model_config

DEFAULT_OUT = Path("outputs") / "finetune"
REPO_FINETUNE_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "finetune" / "finetune.yaml"
)


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, event: str, **kw) -> None:
        rec = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **kw,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(json.dumps(rec, default=float), flush=True)


def _options(o) -> dict:
    """JSON-friendly copy of the CLI options (drops the dispatch handler)."""
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(o).items()
        if not callable(v)
    }


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _plain_state_dict(module) -> dict:
    """State dict without Fabric / DDP wrapper prefixes."""
    sd = module.state_dict()
    for prefix in ("_forward_module.", "module."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix) :]: v for k, v in sd.items()}
    return {k: v.detach().cpu() for k, v in sd.items()}


class EarlyStopping:
    def __init__(self, patience: int, mode: str):
        self.patience, self.mode = patience, mode
        self.best, self.best_epoch, self.bad = None, None, 0

    def update(self, value: float, epoch: int) -> bool:
        """Return True if ``value`` is a new best."""
        better = self.best is None or (
            value < self.best if self.mode == "min" else value > self.best
        )
        if better:
            self.best, self.best_epoch, self.bad = value, epoch, 0
        else:
            self.bad += 1
        return better

    @property
    def stop(self) -> bool:
        return self.bad >= self.patience


# ------------------------------------------------------------------------- CTA


def _normalized_copy(src: str, dst: Path, pre: dict) -> Path:
    import scanpy as sc

    a = sc.read_h5ad(src)
    if pre.get("normalize_total"):
        sc.pp.normalize_total(a, target_sum=float(pre["normalize_total"]))
    if pre.get("log1p"):
        sc.pp.log1p(a)
    a.obs_names_make_unique()
    a.write_h5ad(dst)
    return dst


def _evaluate_cta(model, loader, n_classes: int) -> dict:
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    loss, ys, ps = 0.0, [], []
    with torch.no_grad():
        for b in loader:
            out = model(
                gene_ids=b["gene"],
                value_log1p=b["masked_continuous_expr"],
                value_binned=b["masked_discrete_expr"],
            )
            out = out[0] if isinstance(out, tuple) else out
            loss += ce(out.float(), b["cell_type_id"]).item()
            ys.append(b["cell_type_id"].cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {
        "loss": loss / len(y),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(
            f1_score(y, p, average="macro", labels=range(n_classes), zero_division=0)
        ),
        "cells": int(len(y)),
    }


def finetune_cta(model_cfg, dataset_cfg, run_dir: Path, log: RunLog, o) -> dict:
    import scanpy as sc
    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy
    from omegaconf import OmegaConf

    from deepsc.finetune.cell_type_annotation import CellTypeAnnotation

    pre = {"normalize_total": 10000, "log1p": True, "max_length": o.max_length}
    col = dataset_cfg.get("celltype_col", "celltype")
    data_dir = run_dir / "_data"
    data_dir.mkdir(exist_ok=True)
    train_h5 = _normalized_copy(
        dataset_cfg["data"]["train"], data_dir / "train.h5ad", pre
    )
    val_h5 = _normalized_copy(dataset_cfg["data"]["val"], data_dir / "val.h5ad", pre)
    labels = [
        set(sc.read_h5ad(p, backed="r").obs[col].astype(str))
        for p in (train_h5, val_h5)
    ]
    classes = sorted(labels[0] & labels[1])  # same rule as CellTypeAnnotation
    log("data_ready", classes=classes, normalization=pre)

    base = OmegaConf.load(REPO_FINETUNE_CONFIG)
    for k in ("defaults", "hydra"):
        base.pop(k, None)
    a = model_cfg["architecture"]
    args = OmegaConf.merge(
        base,
        {
            "model": {"num_genes": a["num_genes"], "num_bins": a["num_bins"]},
            "pretrained_model_path": model_cfg["checkpoint"],
            "load_pretrained_model": True,
            "csv_path": model_cfg["gene_map"],
            "data_path": str(train_h5),
            "data_path_eval": str(val_h5),
            "seperated_train_eval_dataset": True,
            "var_name_in_h5ad": dataset_cfg.get("var_name_col"),
            "obs_celltype_col": col,
            "epoch": o.epochs,
            "batch_size": o.batch_size,
            "grad_acc": 1,
            "learning_rate": o.lr,
            "sequence_length": o.max_length,
            "seed": o.seed,
            "num_device": 1,
            "finetune_mode": "full",
            "lr_scheduler_type": "constant",
            "cls_loss_type": o.cls_loss,
            "resume_last_training": False,
        },
    )
    # CellTypeAnnotation uses DistributedSampler, which needs a process group: single-device DDP.
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        strategy=DDPStrategy(
            find_unused_parameters=True, gradient_as_bucket_view=False
        ),
        precision="bf16-mixed",
    )
    fabric.launch()
    model = models.build_cta_model(model_cfg, len(classes))
    with contextlib.chdir(run_dir):
        trainer = CellTypeAnnotation(args, fabric=fabric, model=model)
    if trainer.common_celltypes != classes:
        raise RuntimeError(
            f"class order mismatch: {trainer.common_celltypes} != {classes}"
        )

    stopper, history = EarlyStopping(o.patience, "min"), []
    for epoch in range(1, o.epochs + 1):
        t0 = time.time()
        trainer.model.train()
        losses = []
        for step, data in enumerate(trainer.train_loader):
            loss, _ = trainer.each_training_iteration(data, is_accumulating=False)
            losses.append(loss.item())
            if step % 200 == 0:
                log(
                    "train_step",
                    epoch=epoch,
                    step=step,
                    total_steps=len(trainer.train_loader),
                    loss=losses[-1],
                )
            if o.limit_steps and step + 1 >= o.limit_steps:
                break
        val = _evaluate_cta(trainer.model, trainer.eval_loader, len(classes))
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": val,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        log("epoch_complete", **rec)
        if stopper.update(val["loss"], epoch):
            import torch

            torch.save(
                {
                    "model": _plain_state_dict(trainer.model),
                    "classes": classes,
                    "epoch": epoch,
                    "val": val,
                    "args": _options(o),
                },
                run_dir / "best_model.pt",
            )
            log("best_saved", epoch=epoch, val_loss=val["loss"])
        if stopper.stop:
            log("early_stop", epoch=epoch)
            break
    for p in (train_h5, val_h5):
        p.unlink(missing_ok=True)
    data_dir.rmdir()
    return {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "classes": classes,
        "selection": "min validation cross-entropy",
        "head": {"num_layers_cls": 3, "cell_emb_style": "avg-pool"},
        "preprocessing": pre,
    }


# -------------------------------------------------------------------------- PP


def _pp_val_metrics(
    model, loader, gene_ids, genes, adata, ctrl, de_idx, device
) -> dict:
    """Mapped-gene metrics on a GEARS loader: Pearson Δ (all / top-20 DE), MSE, control-mean MSE."""
    import torch
    from scipy.stats import pearsonr

    model.eval()
    mk = gene_ids > 0
    sums, counts, sq, sq_ctrl, n = {}, {}, 0.0, 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in loader:
            b.to(device)
            p = (
                model.pred_perturb(
                    b, include_zero_gene="all", gene_ids=gene_ids, genes=genes
                )
                .float()
                .cpu()
                .numpy()
            )
            t = b.y.float().cpu().numpy()
            sq += float(((p[:, mk] - t[:, mk]) ** 2).sum())
            sq_ctrl += float(((ctrl[mk][None] - t[:, mk]) ** 2).sum())
            n += len(t)
            for c, pi, ti in zip(b.pert, p, t):
                s = sums.setdefault(c, [0.0, 0.0])
                s[0], s[1] = s[0] + pi, s[1] + ti
                counts[c] = counts.get(c, 0) + 1
    r_all, r_de = [], []
    for c, (ps, ts) in sums.items():
        dp, dt = ps / counts[c] - ctrl, ts / counts[c] - ctrl
        r_all.append(pearsonr(dp[mk], dt[mk])[0])
        de = de_idx[c][mk[de_idx[c]]]
        r_de.append(pearsonr(dp[de], dt[de])[0])
    m = int(mk.sum())
    return {
        "pearson_delta": float(np.mean(r_all)),
        "pearson_de_delta": float(np.mean(r_de)),
        "mse": sq / (n * m),
        "control_baseline_mse": sq_ctrl / (n * m),
        "cells": n,
        "conditions": len(sums),
    }


def finetune_pp(model_cfg, dataset_cfg, run_dir: Path, log: RunLog, o) -> dict:
    import torch
    from gears import PertData

    from deepsc.models.scgpt_pert import map_raw_id_to_vocab_id, masked_mse_loss
    from deepsc.viz.perturbation import top_de_indices

    device = "cuda"
    d = dataset_cfg["data"]
    pert = PertData(d["gears_root"])
    pert.load(data_name=d["gears_name"])
    pert.prepare_split(
        split=d["split_name"],
        seed=d["split_seed"],
        train_gene_set_size=d["train_gene_set_size"],
    )
    pert.get_dataloader(batch_size=o.batch_size, test_batch_size=o.eval_batch_size)
    genes = pert.adata.var["gene_name"].astype(str).tolist()
    n_genes = len(genes)
    vocab = models.gene_map(model_cfg)
    gene_ids = np.array([vocab[g] + 1 if g in vocab else 0 for g in genes], dtype=int)
    name2col = {g: i for i, g in enumerate(genes)}
    X = pert.adata.X
    oc = pert.adata.obs["condition"].to_numpy()
    ctrl = np.asarray(X[oc == "ctrl"].mean(0), dtype=np.float64).ravel()
    val_conds = sorted({c for b in pert.dataloader["val_loader"] for c in b.pert})
    de_idx = top_de_indices(pert.adata, val_conds)
    log(
        "data_ready",
        genes=n_genes,
        mapped_genes=int((gene_ids > 0).sum()),
        val_conditions=len(val_conds),
    )

    model = models.build_pp_model(model_cfg)
    state, _ = models.load_state(model_cfg["checkpoint"])
    info = model.load_state_dict(state, strict=False)
    log(
        "pretrained_loaded",
        missing_keys=list(info.missing_keys),
        unexpected_keys=list(info.unexpected_keys),
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=o.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
    vocab_ids = torch.as_tensor(gene_ids, device=device)

    stopper, history = EarlyStopping(o.patience, "max"), []
    loader = pert.dataloader["train_loader"]
    for epoch in range(1, o.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for step, b in enumerate(loader):
            bs = len(b.y)
            b.to(device)
            x = b.x[:, 0].view(bs, n_genes)
            flags = torch.zeros(bs, n_genes, device=device, dtype=torch.long)
            for r_, p_ in enumerate(b.pert):
                for g in p_.split("+"):
                    if g and g != "ctrl" and g in name2col:
                        flags[r_, name2col[g]] = 1
            idx = torch.arange(n_genes, device=device)
            if n_genes > o.max_length:
                idx = torch.randperm(n_genes, device=device)[: o.max_length]
            mapped_ids = map_raw_id_to_vocab_id(idx, gene_ids).repeat(bs, 1)
            valid = (vocab_ids[idx] != 0).float().unsqueeze(0)
            inp, tgt = x[:, idx] * valid, b.y[:, idx] * valid
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _, _ = model(mapped_ids, inp, flags[:, idx])
                loss = masked_mse_loss(
                    out.float(), tgt, torch.ones_like(inp, dtype=torch.bool)
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if step % 500 == 0:
                log(
                    "train_step",
                    epoch=epoch,
                    step=step,
                    total_steps=len(loader),
                    loss=losses[-1],
                )
            if o.limit_steps and step + 1 >= o.limit_steps:
                break
        val = _pp_val_metrics(
            model,
            pert.dataloader["val_loader"],
            gene_ids,
            genes,
            pert.adata,
            ctrl,
            de_idx,
            device,
        )
        sched.step()
        rec = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "val": val,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        log("epoch_complete", **rec)
        if stopper.update(val["pearson_delta"], epoch):
            torch.save(
                {
                    "model": _plain_state_dict(model),
                    "epoch": epoch,
                    "val": val,
                    "args": _options(o),
                },
                run_dir / "best_model.pt",
            )
            log("best_saved", epoch=epoch, val_pearson_delta=val["pearson_delta"])
        if stopper.stop:
            log("early_stop", epoch=epoch)
            break
    return {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "metric_convention": "mapped_genes",
        "selection": "max validation Pearson delta (mapped genes)",
        "preprocessing": {"include_zero_gene": "all"},
    }


# ------------------------------------------------------------------------- run

FINETUNE = {"cta": finetune_cta, "pp": finetune_pp}


def run(o) -> Path:
    import yaml

    dataset_cfg = load_dataset_config(o.task, o.dataset)
    model_cfg = load_model_config(o.model)
    if model_cfg["type"] != "pretrained":
        raise ValueError(
            f"{model_cfg['id']} is {model_cfg['type']!r}; fine-tune a pre-trained model"
        )
    digest = models.verify_checkpoint(model_cfg)
    run_id = (
        o.run_id
        or f"{model_cfg['id'].replace('_pretrain', '')}_{o.task}_{dataset_cfg['name']}_"
        f"{datetime.now().strftime('%Y%m%dT%H%M')}"
    )
    run_dir = (Path(o.out) / o.task / dataset_cfg["name"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    log = RunLog(run_dir / "train.log")
    _seed(o.seed)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log(
        "start",
        task=o.task,
        dataset=dataset_cfg["name"],
        model=model_cfg["id"],
        run_dir=str(run_dir),
        device=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    result = FINETUNE[o.task](model_cfg, dataset_cfg, run_dir, log, o)

    status = {
        "status": "completed",
        "task": o.task,
        "dataset": dataset_cfg["name"],
        "run_id": run_id,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": code_version(),
        "config": {
            **_options(o),
            "checkpoint": model_cfg["checkpoint"],
            "checkpoint_sha256": digest,
        },
        **{k: result[k] for k in ("history", "best_epoch", "selection") if k in result},
        **({"classes": result["classes"]} if "classes" in result else {}),
        **(
            {"metric_convention": result["metric_convention"]}
            if "metric_convention" in result
            else {}
        ),
    }
    (run_dir / "status.json").write_text(
        json.dumps(status, indent=2, default=float) + "\n"
    )
    best = run_dir / "best_model.pt"
    model_yaml = {
        "id": run_id,
        "type": "finetuned",
        "base": model_cfg["id"],
        "task": o.task,
        "dataset": dataset_cfg["name"],
        "checkpoint": str(best.resolve()),
        "sha256": models.sha256(best),
        "training_record": str((run_dir / "status.json").resolve()),
        "preprocessing": result["preprocessing"],
        **({"head": result["head"]} if "head" in result else {}),
    }
    (run_dir / "model.yaml").write_text(
        f"# Generated by `deepsc finetune` on {status['completed_at']}.\n"
        + yaml.safe_dump(model_yaml, sort_keys=False)
    )
    for sub in (
        "checkpoints",
        "logs",
        "visualizations",
    ):  # created empty by CellTypeAnnotation
        with contextlib.suppress(OSError):
            (run_dir / sub).rmdir()
    log(
        "done",
        best_epoch=result["best_epoch"],
        model_config=str(run_dir / "model.yaml"),
    )
    return run_dir


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, choices=sorted(FINETUNE))
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--model", required=True, help="pre-trained model config id or YAML path"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="writes <out>/<task>/<dataset>/<run_id>/",
    )
    p.add_argument("--run-id", help="default: <model>_<task>_<dataset>_<timestamp>")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, help="default: 16 (cta), 8 (pp)")
    p.add_argument(
        "--eval-batch-size", type=int, default=4, help="pp validation batch size"
    )
    p.add_argument(
        "--max-length",
        type=int,
        help="genes per cell in training; default: 1024 (cta), 1536 (pp)",
    )
    p.add_argument(
        "--cls-loss",
        default="standard",
        choices=["standard", "per_bin"],
        help="cta loss: standard cross-entropy or the class-averaged variant",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit-steps",
        type=int,
        default=0,
        help="stop each epoch after N steps (smoke tests)",
    )


def main_from_args(args) -> None:
    if args.batch_size is None:
        args.batch_size = {"cta": 16, "pp": 8}[args.task]
    if args.max_length is None:
        args.max_length = {"cta": 1024, "pp": 1536}[args.task]
    run_dir = run(args)
    print(
        f"Wrote {run_dir}\nNext: deepsc infer --task {args.task} --dataset {args.dataset} "
        f"--model {run_dir / 'model.yaml'}"
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="deepsc finetune", description=__doc__.split("\n\n")[0]
    )
    add_arguments(p)
    main_from_args(p.parse_args(argv))


if __name__ == "__main__":
    main()
