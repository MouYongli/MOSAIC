"""Prediction contract: the file format that connects ``infer`` and ``report``.

A prediction set is a directory with two files::

    <dir>/predictions.npz   arrays (see SCHEMAS)
    <dir>/meta.json         task, dataset, split, model, preprocessing, provenance

``report`` reads only this format, so any model (pre-trained, fine-tuned, a
baseline or an external tool) can be evaluated once it writes it.

CTA arrays (one row per cell):
    cell_ids  str  (n,)    obs names of the evaluated cells, in h5ad order
    classes   str  (k,)    class codes; index = model output index
    y_true    int  (n,)    index into ``classes``; -1 if the label is not a model class
    logits    f32  (n, k)  raw classifier scores; prediction = argmax

PP arrays (one row per perturbed test cell):
    conditions   str  (n,)    perturbation condition, e.g. "SAMM50+ctrl"
    genes        str  (g,)    gene symbols, same order as the dataset
    predictions  f32  (n, g)  predicted expression after perturbation
    targets      f32  (n, g)  observed expression
    gene_mask    bool (g,)    genes the model can represent; metrics use these only
"""

from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from datetime import datetime, timezone

SCHEMA_VERSION = 1
PREDICTIONS_FILE = "predictions.npz"
META_FILE = "meta.json"

# name -> (dtype kind(s), number of dims)
SCHEMAS: dict[str, dict[str, tuple[str, int]]] = {
    "cta": {
        "cell_ids": ("U", 1),
        "classes": ("U", 1),
        "y_true": ("i", 1),
        "logits": ("f", 2),
    },
    "pp": {
        "conditions": ("U", 1),
        "genes": ("U", 1),
        "predictions": ("f", 2),
        "targets": ("f", 2),
        "gene_mask": ("b", 1),
    },
}


class ContractError(ValueError):
    """Raised when a prediction set does not follow the contract."""


def code_version() -> str | None:
    try:
        root = Path(__file__).resolve().parent
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", "-C", str(root), *a], capture_output=True, text=True, check=True
        ).stdout.strip()
        return run("rev-parse", "--short", "HEAD") + (
            "-dirty" if run("status", "--porcelain") else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class PredictionSet:
    task: str
    arrays: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ checks
    def validate(self) -> "PredictionSet":
        if self.task not in SCHEMAS:
            raise ContractError(
                f"unknown task {self.task!r}; expected one of {sorted(SCHEMAS)}"
            )
        schema = SCHEMAS[self.task]
        missing = set(schema) - set(self.arrays)
        if missing:
            raise ContractError(f"{self.task}: missing arrays {sorted(missing)}")
        for name, (kind, ndim) in schema.items():
            a = self.arrays[name]
            if a.dtype.kind not in kind or a.ndim != ndim:
                raise ContractError(
                    f"{self.task}.{name}: expected kind {kind!r} ndim {ndim}, got {a.dtype} {a.shape}"
                )
        a = self.arrays
        if self.task == "cta":
            n, k = len(a["cell_ids"]), len(a["classes"])
            if a["y_true"].shape != (n,) or a["logits"].shape != (n, k):
                raise ContractError(
                    f"cta: shapes disagree (cells {n}, classes {k}, logits {a['logits'].shape})"
                )
            if a["y_true"].min(initial=0) < -1 or a["y_true"].max(initial=0) >= k:
                raise ContractError("cta: y_true out of range")
        else:
            n, g = len(a["conditions"]), len(a["genes"])
            for name in ("predictions", "targets"):
                if a[name].shape != (n, g):
                    raise ContractError(
                        f"pp: {name} shape {a[name].shape} != ({n}, {g})"
                    )
            if a["gene_mask"].shape != (g,):
                raise ContractError("pp: gene_mask length != number of genes")
        for key in ("task", "dataset", "split", "model"):
            if key not in self.meta:
                raise ContractError(f"meta.json missing {key!r}")
        if self.meta["task"] != self.task:
            raise ContractError(f"meta task {self.meta['task']!r} != {self.task!r}")
        return self

    # ------------------------------------------------------------------- views
    @property
    def n(self) -> int:
        return len(self.arrays["cell_ids" if self.task == "cta" else "conditions"])

    @property
    def y_pred(self) -> np.ndarray:
        return self.arrays["logits"].argmax(1)

    # --------------------------------------------------------------------- I/O
    def save(self, out_dir: Path) -> Path:
        self.meta.setdefault("schema_version", SCHEMA_VERSION)
        self.meta.setdefault("task", self.task)
        self.meta.setdefault(
            "created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        self.meta.setdefault("code_version", code_version())
        self.meta["n"] = self.n
        self.validate()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / PREDICTIONS_FILE, **self.arrays)
        (out_dir / META_FILE).write_text(
            json.dumps(self.meta, indent=2, default=str) + "\n"
        )
        return out_dir

    @classmethod
    def load(cls, path: Path) -> "PredictionSet":
        path = Path(path)
        meta = json.loads((path / META_FILE).read_text())
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ContractError(
                f"{path}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}"
            )
        with np.load(path / PREDICTIONS_FILE, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return cls(meta["task"], arrays, meta).validate()
