# Downstream pipeline: finetune → infer → report

The downstream tasks (cell type annotation `cta`, perturbation prediction `pp`)
run in three stages connected by fixed file formats, so each stage can be
swapped or re-run on its own.

| Stage | Command | Status |
|---|---|---|
| Fine-tune a pre-trained model on a benchmark | `deepsc finetune` | available |
| Run a model on a benchmark → prediction set | `deepsc infer` | available |
| Prediction sets → figures, tables, captions | `deepsc report` | available |

Every command is also available as `python -m deepsc <command>`.
`deepsc list` shows the registered datasets and models.

## Configs

- `configs/datasets/<task>/<name>.yaml`: data paths and metadata of a benchmark
  (for CTA also the class code → cell type name mapping).
- `configs/models/<id>.yaml`: architecture, checkpoint (with SHA-256, verified
  before use) and task head. `base: <id>` inherits another model config, so a
  fine-tuned model only lists what differs from its pre-trained base.

How to write them and how to add new ones:
[configs/datasets/README.md](../configs/datasets/README.md),
[configs/models/README.md](../configs/models/README.md).

## Fine-tuning

```bash
deepsc finetune --task cta --dataset zheng   --model deepsc_pretrain_1215
deepsc finetune --task pp  --dataset adamson --model deepsc_pretrain_1215
```

Output: `outputs/finetune/<task>/<dataset>/<run_id>/`:

- `best_model.pt`: checkpoint of the best validation epoch;
- `status.json`: run record (options, per-epoch history, best epoch), used by
  `report` for the training-history figures;
- `model.yaml`: model config of the result, ready for
  `deepsc infer --model outputs/finetune/.../model.yaml`;
- `train.log`: one JSON event per line.

| | CTA | PP |
|---|---|---|
| Training code | `CellTypeAnnotation` (data, model loading, optimizer, step) | training step of `finetune_perturbation.py` |
| Defaults | batch 16, 1,024 genes/cell, Adam 1e-4 constant, cross-entropy | batch 8, 1,536 genes/cell, Adam 1e-4 × 0.9/epoch, masked MSE |
| Preprocessing | TP10K + log1p (same as `infer`) | GEARS processed data |
| Best epoch | lowest validation cross-entropy | highest validation Pearson Δ (mapped genes) |

Both: up to 10 epochs, early stopping after 3 epochs without improvement, bf16,
seed 42. Options: `--epochs`, `--patience`, `--lr`, `--batch-size`,
`--max-length`, `--cls-loss standard|per_bin`, `--seed`, `--run-id`,
`--limit-steps N` (smoke tests). Select the GPU with `CUDA_VISIBLE_DEVICES`.

## Inference

```bash
deepsc infer --task cta --dataset zheng   --model deepsc_cta_zheng_1215
deepsc infer --task pp  --dataset adamson --model deepsc_pp_adamson_1215
```

Output: `outputs/infer/<task>/<dataset>/<model>/<split>/{predictions.npz, meta.json}`.
Options: `--split`, `--device`, `--precision bf16|fp32`, `--batch-size`,
`--limit N` (smoke tests), `--no-hash-check`.

Pre-trained models without a task head are rejected for now (zero-shot, few-shot
and instruction-tuned models are planned).

## Report

```bash
deepsc report --task cta --dataset zheng                     # every model with predictions
deepsc report --task pp  --dataset adamson --models A B      # selected models
```

Reads `outputs/infer/<task>/<dataset>/<model>/<split>/` and writes
`outputs/viz/<task>/<dataset>/`:

- `<model>/`: one untitled figure per panel (PDF + 300 dpi PNG) with a JSON
  sidecar (`title`, `caption_short`, `caption_long`, `description`, `metrics`),
  tables, `manifest.json` and a `README.md` index. Training-history figures are
  added when the model config has a `training_record`.
- `comparison/`: all models side by side (metrics table and figures; PP also
  shows the train-mean-Δ and control-mean baselines).

Each prediction set is checked against the dataset (cell order and labels for
CTA; genes, cells per condition and targets for PP) before plotting; imported
legacy runs must also reproduce their recorded metrics. A failed check stops
the report. A single set can be plotted with `python -m deepsc.viz.cta DIR` or
`python -m deepsc.viz.perturbation DIR`.

## Prediction set format

Defined and validated in `deepsc.pipeline.contract`; `report` reads only this.

- CTA: `cell_ids`, `classes`, `y_true` (index into `classes`, −1 = unknown
  label), `logits`.
- PP: `conditions`, `genes`, `predictions`, `targets`, `gene_mask` (genes the
  model can represent; metrics use these only).
- `meta.json`: task, dataset, split, model (id, checkpoint, SHA-256, lineage),
  preprocessing, runtime, code version.

## Results of the 2026-11-20 external runs

```bash
deepsc import-legacy --task cta --dataset zheng \
    --run-dir /DATA2/DeepSC/results/downstream_20261120/cta_zheng_1215 --model-id deepsc_cta_zheng_1215_legacy
deepsc import-legacy --task pp --dataset adamson \
    --run-dir /DATA2/DeepSC/results/downstream_20261120/pp_adamson_1215 --model-id deepsc_pp_adamson_1215_legacy
```

These runs used a script outside this repository. `deepsc infer` with the same
checkpoints reproduces them as follows (2026-11-25, H100, bf16):

- PP (Adamson, 9,741 cells): identical conditions, genes and targets;
  predictions on mapped genes differ by at most 0.023 (bf16 rounding) and the
  MSE is identical (0.130913).
- CTA (Zheng, 6,606 cells): identical cells and labels; logits agree within
  0.1 for 98.9% of cells and 15 cells (0.23%) get a different class (accuracy
  0.8227 vs 0.8241). The differences come from the 84 cells with more than
  1,023 mapped non-zero genes: MOSAIC keeps the most highly expressed genes,
  while the external script handled long cells differently (not recoverable
  without that script).
