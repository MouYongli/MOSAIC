"""Convert ``test_predictions.npz`` from the 2026-11-20 external runs into prediction sets.

deepsc import-legacy --task cta --dataset zheng   --run-dir /DATA2/DeepSC/results/downstream_20261120/cta_zheng_1215
deepsc import-legacy --task pp  --dataset adamson --run-dir /DATA2/DeepSC/results/downstream_20261120/pp_adamson_1215
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import argparse
from deepsc.pipeline.contract import PredictionSet
from deepsc.pipeline.infer import DEFAULT_OUT
from deepsc.pipeline.registry import load_dataset_config

LEGACY_FILE = "test_predictions.npz"


def convert(task: str, run_dir: Path) -> tuple[dict, dict]:
    z = np.load(Path(run_dir) / LEGACY_FILE)
    status = json.loads((Path(run_dir) / "status.json").read_text())
    if task == "cta":
        arrays = {
            "cell_ids": z["cell_ids"].astype(str),
            "classes": z["classes"].astype(str),
            "y_true": z["labels"].astype(np.int64),
            "logits": z["logits"].astype(np.float32),
        }
    else:
        arrays = {
            "conditions": z["conditions"].astype(str),
            "genes": z["genes"].astype(str),
            "predictions": z["predictions"].astype(np.float32),
            "targets": z["targets"].astype(np.float32),
            "gene_mask": z["gene_vocab_ids"] > 0,
        }
    meta = {
        "legacy": {
            "run_dir": str(run_dir),
            "file": LEGACY_FILE,
            "config": status.get("config"),
            "recorded_test": status.get("test"),
            "best_epoch": status.get("best_epoch"),
        }
    }
    return arrays, meta


def run(
    task: str, dataset: str, run_dir: Path, model_id: str, out: Path = DEFAULT_OUT
) -> Path:
    cfg = load_dataset_config(task, dataset)
    arrays, meta = convert(task, run_dir)
    meta = {
        "task": task,
        "dataset": cfg["name"],
        "split": "test",
        "model": {"id": model_id, "type": "finetuned", "source": "legacy external run"},
        "dataset_config": cfg["_path"],
        **meta,
    }
    out_dir = Path(out) / task / cfg["name"] / model_id / "test"
    PredictionSet(task, arrays, meta).save(out_dir)
    return out_dir


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, choices=["cta", "pp"])
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=f"dir with {LEGACY_FILE} and status.json",
    )
    p.add_argument(
        "--model-id", help="id for the output dir (default: <run-dir name>_legacy)"
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)


def main_from_args(args) -> None:
    model_id = args.model_id or f"{Path(args.run_dir).name}_legacy"
    print(f"Wrote {run(args.task, args.dataset, args.run_dir, model_id, args.out)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="deepsc import-legacy", description=__doc__.split("\n\n")[0]
    )
    add_arguments(p)
    main_from_args(p.parse_args(argv))


if __name__ == "__main__":
    main()
