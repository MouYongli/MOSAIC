# Model configs

One YAML file per model: `configs/models/<id>.yaml`. `deepsc finetune|infer
--model <id>` loads it; an explicit YAML path also works, for example the
`model.yaml` that `deepsc finetune` writes next to each fine-tuned checkpoint.

## Keys

| Key | Meaning |
|---|---|
| `id` | Model id; used in output paths and figure ids. Keep it equal to the file name (not enforced). |
| `type` | `pretrained` (no task head; can be fine-tuned) or `finetuned` (has a task head; can be used for inference). `few_shot` and `instruct` are planned. |
| `base` | Id of another model config. All of its keys are inherited, with nested mappings merged, and this file overrides them. |
| `family` | Model family, currently `deepsc`. |
| `checkpoint` | Weights file. Either `{"model": state_dict, ...}` or the DeepSC pre-training format. |
| `sha256` | SHA-256 of `checkpoint`. Checked before every run; `--no-hash-check` skips the check for `infer`. Compute it with `sha256sum`. |
| `gene_map` | CSV mapping `feature_name` to `id` (0-based; model token = `id + 1`). |
| `architecture` | Constructor arguments of the DeepSC model, including `moe`. |
| `task`, `dataset` | Fine-tuned models only: the task and the dataset the head was trained for. |
| `head` | CTA only: `num_layers_cls` and `cell_emb_style` of the classifier. |
| `preprocessing` | Inference preprocessing. CTA: `normalize_total`, `log1p`, `max_length`. PP: `include_zero_gene`. |
| `training_record` | `status.json` of the fine-tuning run. `deepsc report` uses it for training-history figures. |

Relative paths are resolved against the repository root.

## Example: a fine-tuned model on top of a pre-trained base

```yaml
id: deepsc_cta_zheng_1215
type: finetuned
base: deepsc_pretrain_1215        # inherits architecture and gene_map
task: cta
dataset: zheng
checkpoint: /DATA2/DeepSC/checkpoints/cta_zheng_1215/best_model.pt
sha256: d0a52d4d...
head: {num_layers_cls: 3, cell_emb_style: avg-pool}
preprocessing: {normalize_total: 10000, log1p: true, max_length: 1024}
training_record: /DATA2/DeepSC/results/downstream_20261120/cta_zheng_1215/status.json
```

## Registered models

| Id | Type | Notes |
|---|---|---|
| `deepsc_pretrain_1215` | pretrained | Checkpoint used for the 2026-11-20 runs. |
| `deepsc_cta_zheng_1215` | finetuned | CTA on Zheng68K, trained with the external 2026-11-20 script. |
| `deepsc_pp_adamson_1215` | finetuned | PP on Adamson, trained with the external 2026-11-20 script. |

`deepsc finetune` writes a ready-to-use config to
`outputs/finetune/<task>/<dataset>/<run_id>/model.yaml`. To make such a model
permanent, move the checkpoint to shared storage, copy the YAML here, and
update `checkpoint`, `sha256` and `training_record`.
