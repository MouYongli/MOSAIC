"""Run a model on a benchmark dataset and write a prediction set (see ``contract``).

    deepsc infer --task cta --dataset zheng   --model deepsc_cta_zheng_1215
    deepsc infer --task pp  --dataset adamson --model deepsc_pp_adamson_1215

Output: ``<out>/<task>/<dataset>/<model_id>/<split>/{predictions.npz, meta.json}``
(default ``<out>`` = ``outputs/infer``).
"""

from __future__ import annotations
import contextlib
from pathlib import Path

import numpy as np

import argparse
import time
from deepsc.pipeline import models
from deepsc.pipeline.contract import PredictionSet
from deepsc.pipeline.registry import load_dataset_config, load_model_config

DEFAULT_OUT = Path("outputs") / "infer"


def _autocast(device: str, precision: str):
    import torch

    if precision == "bf16" and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _check_model_for_task(model_cfg: dict, task: str) -> None:
    if model_cfg["type"] == "pretrained":
        raise NotImplementedError(
            f"{model_cfg['id']} is a pre-trained model without a {task} head; zero-shot {task} inference is not "
            "supported yet. Fine-tune it first or pass a fine-tuned model."
        )
    if model_cfg.get("task") != task:
        raise ValueError(
            f"{model_cfg['id']} was fine-tuned for task {model_cfg.get('task')!r}, not {task!r}"
        )


# ------------------------------------------------------------------------- CTA


def infer_cta(
    model_cfg,
    dataset_cfg,
    split="test",
    device="cuda",
    precision="bf16",
    batch_size=32,
    limit=None,
):
    import scanpy as sc
    import torch
    from torch.utils.data import DataLoader, Subset

    from deepsc.data.data_collator import DataCollator
    from deepsc.data.dataset import GeneExpressionDatasetMappedWithGlobalCelltype

    _check_model_for_task(model_cfg, "cta")
    state, extras = models.load_state(model_cfg["checkpoint"])
    classes = [
        str(c) for c in (extras.get("classes") or model_cfg.get("classes") or [])
    ]
    if not classes:
        raise ValueError(
            f"{model_cfg['id']}: class list not found in checkpoint or model config ('classes')"
        )
    model = models.build_cta_model(model_cfg, len(classes))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    adata = sc.read_h5ad(dataset_cfg["data"][split])
    pre = model_cfg.get("preprocessing", {})
    if pre.get("normalize_total"):
        sc.pp.normalize_total(adata, target_sum=float(pre["normalize_total"]))
    if pre.get("log1p"):
        sc.pp.log1p(adata)
    col = dataset_cfg.get("celltype_col", "celltype")
    labels = adata.obs[col].astype(str).to_numpy()
    type2id = {c: i for i, c in enumerate(classes)}
    y_true = np.array([type2id.get(c, -1) for c in labels], dtype=np.int64)
    # The dataset class rejects unknown labels; feed it a placeholder for those cells.
    adata.obs["_label"] = np.where(y_true >= 0, labels, classes[0])
    ds = GeneExpressionDatasetMappedWithGlobalCelltype(
        h5ad=adata,
        csv_path=model_cfg["gene_map"],
        var_name_col=dataset_cfg.get("var_name_col"),
        obs_celltype_col="_label",
        global_type2id=type2id,
        global_id2type={i: c for c, i in type2id.items()},
    )
    a = model_cfg["architecture"]
    collate = DataCollator(
        do_padding=True,
        pad_token_id=0,
        pad_value=0,
        do_mlm=False,
        do_binning=True,
        max_length=pre.get("max_length", 1024),
        num_genes=a["num_genes"],
        num_bins=a["num_bins"],
        use_max_cell_length=False,
        cell_type=True,
    )
    idx = range(len(ds) if limit is None else min(limit, len(ds)))
    loader = DataLoader(
        Subset(ds, idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
    )
    out = []
    with torch.no_grad(), _autocast(device, precision):
        for b in loader:
            o = model(
                gene_ids=b["gene"].to(device),
                value_log1p=b["masked_continuous_expr"].to(device),
                value_binned=b["masked_discrete_expr"].to(device),
            )
            out.append((o[0] if isinstance(o, tuple) else o).float().cpu().numpy())
    n = len(idx)
    arrays = {
        "cell_ids": adata.obs_names.to_numpy().astype(str)[:n],
        "classes": np.array(classes),
        "y_true": y_true[:n],
        "logits": np.concatenate(out).astype(np.float32),
    }
    meta = {
        "preprocessing": pre,
        "unknown_labels": int((y_true[:n] < 0).sum()),
        "inputs": {"h5ad": dataset_cfg["data"][split]},
    }
    return arrays, meta


# -------------------------------------------------------------------------- PP


def infer_pp(
    model_cfg,
    dataset_cfg,
    split="test",
    device="cuda",
    precision="bf16",
    batch_size=4,
    limit=None,
):
    import torch
    from gears import PertData

    _check_model_for_task(model_cfg, "pp")
    d = dataset_cfg["data"]
    pert = PertData(d["gears_root"])
    pert.load(data_name=d["gears_name"])
    pert.prepare_split(
        split=d["split_name"],
        seed=d["split_seed"],
        train_gene_set_size=d["train_gene_set_size"],
    )
    pert.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)
    loader = pert.dataloader[
        {"test": "test_loader", "val": "val_loader", "train": "train_loader"}[split]
    ]

    genes = pert.adata.var["gene_name"].astype(str).tolist()
    vocab = models.gene_map(model_cfg)
    gene_ids = np.array(
        [vocab[g] + 1 if g in vocab else 0 for g in genes], dtype=int
    )  # 0 = not in vocabulary

    state, _ = models.load_state(model_cfg["checkpoint"])
    model = models.build_pp_model(model_cfg)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    pre = model_cfg.get("preprocessing", {})
    preds, targets, conds = [], [], []
    with torch.no_grad(), _autocast(device, precision):
        for batch in loader:
            batch.to(device)
            p = model.pred_perturb(
                batch,
                include_zero_gene=pre.get("include_zero_gene", "all"),
                gene_ids=gene_ids,
                genes=genes,
            )
            preds.append(p.float().cpu().numpy())
            targets.append(batch.y.float().cpu().numpy())
            conds += list(batch.pert)
            if limit is not None and len(conds) >= limit:
                break
    n = len(conds) if limit is None else min(limit, len(conds))
    arrays = {
        "conditions": np.array(conds[:n]),
        "genes": np.array(genes),
        "predictions": np.concatenate(preds)[:n].astype(np.float32),
        "targets": np.concatenate(targets)[:n].astype(np.float32),
        "gene_mask": gene_ids > 0,
    }
    meta = {
        "preprocessing": pre,
        "inputs": {
            "gears_root": d["gears_root"],
            "gears_name": d["gears_name"],
            "split": f"{d['split_name']}/seed{d['split_seed']}",
        },
    }
    return arrays, meta


INFER = {"cta": infer_cta, "pp": infer_pp}


def run(
    task,
    dataset,
    model,
    split="test",
    out=DEFAULT_OUT,
    device="cuda",
    precision="bf16",
    batch_size=None,
    limit=None,
    check_hash=True,
) -> Path:
    dataset_cfg = load_dataset_config(task, dataset)
    model_cfg = load_model_config(model)
    digest = models.verify_checkpoint(model_cfg) if check_hash else None
    t0 = time.time()
    kw = {"batch_size": batch_size} if batch_size else {}
    arrays, meta = INFER[task](
        model_cfg,
        dataset_cfg,
        split=split,
        device=device,
        precision=precision,
        limit=limit,
        **kw,
    )
    meta = {
        "task": task,
        "dataset": dataset_cfg["name"],
        "split": split,
        "model": {
            "id": model_cfg["id"],
            "type": model_cfg["type"],
            "checkpoint": model_cfg["checkpoint"],
            "sha256": digest,
            "lineage": model_cfg.get("_lineage", []),
            "config": model_cfg["_path"],
        },
        "dataset_config": dataset_cfg["_path"],
        "runtime": {
            "device": device,
            "precision": precision,
            "seconds": round(time.time() - t0, 1),
            "limit": limit,
        },
        **meta,
    }
    out_dir = Path(out) / task / dataset_cfg["name"] / model_cfg["id"] / split
    PredictionSet(task, arrays, meta).save(out_dir)
    return out_dir


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, choices=sorted(INFER))
    p.add_argument(
        "--dataset",
        required=True,
        help="dataset config name (configs/datasets/<task>/) or YAML path",
    )
    p.add_argument(
        "--model", required=True, help="model config id (configs/models/) or YAML path"
    )
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="root dir; writes <out>/<task>/<dataset>/<model>/<split>",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--batch-size", type=int)
    p.add_argument("--limit", type=int, help="only the first N cells (smoke tests)")
    p.add_argument(
        "--no-hash-check",
        dest="check_hash",
        action="store_false",
        help="skip verifying the checkpoint SHA-256 against the model config",
    )


def main_from_args(args) -> None:
    out = run(
        args.task,
        args.dataset,
        args.model,
        args.split,
        args.out,
        args.device,
        args.precision,
        args.batch_size,
        args.limit,
        args.check_hash,
    )
    print(f"Wrote {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="deepsc infer", description=__doc__.split("\n\n")[0]
    )
    add_arguments(p)
    main_from_args(p.parse_args(argv))


if __name__ == "__main__":
    main()
