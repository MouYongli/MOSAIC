"""Dataset and model registries backed by YAML files under ``configs/``.

* ``configs/datasets/<task>/<name>.yaml``: data paths and dataset metadata.
* ``configs/models/<id>.yaml``: a model = architecture + checkpoint (+ task head).
  ``base: <id>`` inherits every key of another model config (deep merge), so a
  fine-tuned model only states what differs from its pre-trained base.

Relative paths in configs are resolved against the repository root.
"""

from __future__ import annotations
from pathlib import Path

import copy

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs"
DATASET_ROOT = CONFIG_ROOT / "datasets"
MODEL_ROOT = CONFIG_ROOT / "models"
TASKS = ("cta", "pp")
MODEL_TYPES = ("pretrained", "finetuned")  # planned: "few_shot", "instruct"


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(path_or_name: str | Path, root: Path) -> Path:
    path = Path(path_or_name)
    if path.suffix in (".yaml", ".yml"):
        return path
    return root / f"{path_or_name}.yaml"


def _available(root: Path) -> list[str]:
    return sorted(p.stem for p in root.glob("*.yaml"))


def resolve_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = (
            deep_merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


# ------------------------------------------------------------------- datasets


def load_dataset_config(task: str, dataset: str | Path) -> dict:
    """Load ``configs/datasets/<task>/<dataset>.yaml`` (or an explicit YAML path)."""
    path = _resolve(dataset, DATASET_ROOT / task)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset config {path}; available for {task}: {_available(DATASET_ROOT / task)}"
        )
    cfg = _read_yaml(path)
    if cfg.get("task") != task:
        raise ValueError(f"{path} is a {cfg.get('task')!r} config, expected {task!r}")
    cfg["_path"] = str(path)
    return cfg


def list_datasets(task: str | None = None) -> dict[str, list[str]]:
    return {t: _available(DATASET_ROOT / t) for t in ([task] if task else TASKS)}


# --------------------------------------------------------------------- models


def load_model_config(model: str | Path, _seen: tuple = ()) -> dict:
    """Load a model config, resolving ``base`` inheritance and relative paths."""
    path = _resolve(model, MODEL_ROOT)
    if not path.exists():
        raise FileNotFoundError(
            f"No model config {path}; available: {_available(MODEL_ROOT)}"
        )
    cfg = _read_yaml(path)
    key = str(path.resolve())
    if key in _seen:
        raise ValueError(f"circular model base: {' -> '.join(_seen + (key,))}")
    base = cfg.pop("base", None)
    if base:
        parent = load_model_config(base, _seen + (key,))
        parent.pop("_path", None)
        lineage = parent.pop("_lineage", [])
        cfg = deep_merge(parent, cfg)
        cfg["_lineage"] = lineage + [parent["id"]]
    cfg.setdefault("id", path.stem)
    if cfg.get("type") not in MODEL_TYPES:
        raise ValueError(
            f"{path}: type must be one of {MODEL_TYPES}, got {cfg.get('type')!r}"
        )
    for key in ("checkpoint", "gene_map", "training_record"):
        if cfg.get(key):
            cfg[key] = str(resolve_path(cfg[key]))
    cfg["_path"] = str(path)
    return cfg


def list_models() -> list[str]:
    return _available(MODEL_ROOT)
