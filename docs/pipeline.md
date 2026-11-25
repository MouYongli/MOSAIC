# Downstream pipeline: finetune → infer → report

The downstream tasks (cell type annotation `cta`, perturbation prediction `pp`)
run in three stages connected by fixed file formats, so each stage can be
swapped or re-run on its own.

| Stage | Command | Status |
|---|---|---|
| Fine-tune a model on a benchmark | `deepsc finetune` | planned (step 3; use `src/deepsc/finetune/*` for now) |
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
