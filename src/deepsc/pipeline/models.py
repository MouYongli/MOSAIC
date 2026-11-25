"""Build DeepSC models from a model config and load their checkpoints."""

from __future__ import annotations
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(cfg: dict) -> str:
    """Return the checkpoint's SHA-256; raise if the config records a different one."""
    digest = sha256(cfg["checkpoint"])
    if cfg.get("sha256") and cfg["sha256"] != digest:
        raise ValueError(
            f"{cfg['checkpoint']}: sha256 {digest} != config {cfg['sha256']}"
        )
    return digest


def load_state(path: str | Path) -> tuple[dict, dict]:
    """(state_dict, extras) for checkpoints saved as ``{"model": ...}`` or in DeepSC pre-training format."""
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        return raw["model"], {k: v for k, v in raw.items() if k != "model"}
    from deepsc.utils import extract_state_dict

    return extract_state_dict(raw), {}


def gene_map(cfg: dict) -> dict[str, int]:
    """feature_name -> 0-based vocabulary id from the model's gene map CSV."""
    df = pd.read_csv(cfg["gene_map"]).dropna(subset=["feature_name", "id"])
    return dict(zip(df["feature_name"].astype(str), df["id"].astype(int)))


def _moe(arch: dict) -> SimpleNamespace:
    return SimpleNamespace(**arch["moe"])


def build_cta_model(cfg: dict, n_classes: int):
    """DeepSC encoder + classification head (``deepsc.models.deepsc``)."""
    from deepsc.models.deepsc.model import DeepSC, DeepSCClassifier

    a = cfg["architecture"]
    enc_kwargs = {k: v for k, v in a.items() if k != "moe"}
    encoder = DeepSC(
        **enc_kwargs, enable_l0=False, enable_mse=True, enable_ce=True, moe=_moe(a)
    )
    head = cfg.get("head", {})
    return DeepSCClassifier(
        encoder,
        n_cls=n_classes,
        num_layers_cls=head.get("num_layers_cls", 3),
        cell_emb_style=head.get("cell_emb_style", "avg-pool"),
    )


def build_pp_model(cfg: dict):
    """DeepSC perturbation model (``deepsc.models.generation_deepsc``)."""
    from deepsc.models.generation_deepsc.model import DeepSC

    a = cfg["architecture"]
    kwargs = {k: v for k, v in a.items() if k != "moe"}
    return DeepSC(
        **kwargs, enable_l0=False, enable_mse=True, enable_ce=False, moe=_moe(a)
    )
