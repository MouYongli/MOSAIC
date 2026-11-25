"""Unit tests for deepsc.pipeline (contract, registry, CLI); no GPU or /DATA2 needed."""

import json

import numpy as np
import pytest

from deepsc.pipeline import registry
from deepsc.pipeline.contract import ContractError, PredictionSet


def _cta(n=5, k=3):
    rng = np.random.default_rng(0)
    arrays = {
        "cell_ids": np.array([f"c{i}" for i in range(n)]),
        "classes": np.array(["0", "1", "2"][:k]),
        "y_true": rng.integers(0, k, n),
        "logits": rng.normal(size=(n, k)).astype(np.float32),
    }
    meta = {"task": "cta", "dataset": "toy", "split": "test", "model": {"id": "m"}}
    return PredictionSet("cta", arrays, meta)


def _pp(n=4, g=6):
    rng = np.random.default_rng(1)
    arrays = {
        "conditions": np.array(["A+ctrl"] * n),
        "genes": np.array([f"G{i}" for i in range(g)]),
        "predictions": rng.normal(size=(n, g)).astype(np.float32),
        "targets": rng.normal(size=(n, g)).astype(np.float32),
        "gene_mask": np.arange(g) < g - 1,
    }
    meta = {"task": "pp", "dataset": "toy", "split": "test", "model": {"id": "m"}}
    return PredictionSet("pp", arrays, meta)


@pytest.mark.parametrize("make", [_cta, _pp])
def test_roundtrip(tmp_path, make):
    ps = make()
    ps.save(tmp_path / "set")
    back = PredictionSet.load(tmp_path / "set")
    assert (
        back.task == ps.task
        and back.meta["n"] == ps.n
        and back.meta["schema_version"] == 1
    )
    for k, v in ps.arrays.items():
        np.testing.assert_array_equal(back.arrays[k], v)


def test_cta_y_pred_is_argmax():
    ps = _cta()
    np.testing.assert_array_equal(ps.y_pred, ps.arrays["logits"].argmax(1))


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda a, m: a.pop("logits"), "missing arrays"),
        (lambda a, m: a.__setitem__("logits", a["logits"][:, :2]), "shapes disagree"),
        (lambda a, m: a.__setitem__("y_true", a["y_true"] + 5), "out of range"),
        (lambda a, m: m.pop("model"), "missing 'model'"),
        (
            lambda a, m: a.__setitem__("y_true", a["y_true"].astype(float)),
            "expected kind",
        ),
    ],
)
def test_cta_validation_errors(mutate, msg):
    ps = _cta()
    mutate(ps.arrays, ps.meta)
    with pytest.raises(ContractError, match=msg):
        ps.validate()


def test_pp_validation_errors():
    ps = _pp()
    ps.arrays["gene_mask"] = ps.arrays["gene_mask"][:-1]
    with pytest.raises(ContractError, match="gene_mask"):
        ps.validate()


def test_load_rejects_other_schema_version(tmp_path):
    _cta().save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["schema_version"] = 99
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ContractError, match="schema_version"):
        PredictionSet.load(tmp_path)


def test_model_config_inherits_base_and_resolves_paths():
    cfg = registry.load_model_config("deepsc_cta_zheng_1215")
    base = registry.load_model_config("deepsc_pretrain_1215")
    assert cfg["architecture"] == base["architecture"] and cfg["_lineage"] == [
        "deepsc_pretrain_1215"
    ]
    assert cfg["task"] == "cta" and cfg["checkpoint"].endswith("best_model.pt")
    assert cfg["gene_map"] == str(
        registry.REPO_ROOT / "resources/gene_maps/gene_map_tp10k.csv"
    )


def test_model_config_rejects_bad_type_and_cycles(tmp_path):
    (tmp_path / "a.yaml").write_text(
        f"id: a\ntype: finetuned\nbase: {tmp_path / 'b.yaml'}\n"
    )
    (tmp_path / "b.yaml").write_text(
        f"id: b\ntype: finetuned\nbase: {tmp_path / 'a.yaml'}\n"
    )
    with pytest.raises(ValueError, match="circular"):
        registry.load_model_config(tmp_path / "a.yaml")
    (tmp_path / "c.yaml").write_text("id: c\ntype: zero_shot\n")
    with pytest.raises(ValueError, match="type must be"):
        registry.load_model_config(tmp_path / "c.yaml")


def test_deep_merge_keeps_nested_keys():
    assert registry.deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}}) == {
        "a": {"x": 1, "y": 3},
        "b": 1,
    }


def test_cli_lists_registered_configs(capsys):
    from deepsc.cli import main

    main(["list"])
    out = capsys.readouterr().out
    assert "datasets/cta: zheng" in out and "deepsc_pp_adamson_1215" in out


def test_infer_refuses_pretrained_model_without_head():
    from deepsc.pipeline.infer import _check_model_for_task

    with pytest.raises(NotImplementedError, match="zero-shot"):
        _check_model_for_task(registry.load_model_config("deepsc_pretrain_1215"), "cta")
    with pytest.raises(ValueError, match="fine-tuned for task 'pp'"):
        _check_model_for_task(
            registry.load_model_config("deepsc_pp_adamson_1215"), "cta"
        )


def test_report_discover_finds_and_selects_prediction_sets(tmp_path):
    from deepsc.pipeline.report import discover

    for m in ("m1", "m2"):
        ps = _cta()
        ps.meta["model"] = {"id": m}
        ps.save(tmp_path / "cta" / "toy" / m / "test")
    assert [p.parent.name for p in discover("cta", "toy", tmp_path, "test", None)] == [
        "m1",
        "m2",
    ]
    assert [
        p.parent.name for p in discover("cta", "toy", tmp_path, "test", ["m2"])
    ] == ["m2"]
    with pytest.raises(FileNotFoundError, match="m3"):
        discover("cta", "toy", tmp_path, "test", ["m3"])
    with pytest.raises(FileNotFoundError, match="deepsc infer"):
        discover("cta", "toy", tmp_path, "val", None)


def test_early_stopping_min_and_max():
    from deepsc.pipeline.finetune import EarlyStopping

    s = EarlyStopping(patience=2, mode="min")
    assert [s.update(v, e) for e, v in enumerate([1.0, 0.8, 0.9, 0.85], 1)] == [
        True,
        True,
        False,
        False,
    ]
    assert s.best_epoch == 2 and s.stop
    s = EarlyStopping(patience=1, mode="max")
    assert s.update(0.5, 1) and s.update(0.6, 2) and not s.update(0.6, 3) and s.stop


def test_plain_state_dict_strips_wrapper_prefix():
    import torch

    from deepsc.pipeline.finetune import _plain_state_dict

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = torch.nn.Linear(2, 2)

    assert set(_plain_state_dict(Wrapped())) == {"weight", "bias"}
    assert set(_plain_state_dict(torch.nn.Linear(2, 2))) == {"weight", "bias"}


def test_finetune_rejects_non_pretrained_model():
    import argparse
    from deepsc.pipeline.finetune import run

    o = argparse.Namespace(task="cta", dataset="zheng", model="deepsc_cta_zheng_1215")
    with pytest.raises(ValueError, match="fine-tune a pre-trained model"):
        run(o)
