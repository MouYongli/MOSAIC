# Dataset configs

One YAML file per benchmark: `configs/datasets/<task>/<name>.yaml`, where
`<task>` is `cta` (cell type annotation) or `pp` (perturbation prediction).
`deepsc finetune|infer|report --dataset <name>` loads it; an explicit YAML path
also works. `deepsc list` shows the registered configs.

## Common keys

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Short id; used in output paths (`outputs/<stage>/<task>/<name>/`). Keep it equal to the file name (not enforced). |
| `label` | no | Display name in figures and captions (default: `name`). |
| `task` | yes | `cta` or `pp`; must match the directory. |
| `data` | yes | Input files, see below. Absolute paths, or relative to the repository root. |

## CTA (`configs/datasets/cta/`)

```yaml
name: zheng
label: Zheng68K
task: cta
celltype_col: celltype     # obs column with the reference label
var_name_col: null         # var column with gene symbols; null = var_names
data:
  train: /path/zheng_train.h5ad   # raw counts; normalization is applied by the pipeline
  val:   /path/zheng_val.h5ad
  test:  /path/zheng_test.h5ad
classes:                   # optional: label code -> names
  "0": {name: "CD14+ Monocyte", short: "CD14+ Mono", evidence: "LYZ, S100A8, CD14, CST3"}
```

- `classes` is only for display. Figures use `short` and long captions list
  `short = name`. Without it, labels are shown as `Type <code>`.
- If you add `classes`, cover every label code: `deepsc report` checks this.
  Record in `evidence` how the mapping was established (see `zheng.yaml`,
  where the codes were verified against marker genes and class sizes).
- Fine-tuning uses the labels present in both `train` and `val`.

## PP (`configs/datasets/pp/`)

```yaml
name: adamson
label: Adamson
task: pp
data:
  h5ad: /path/adamson/perturb_processed.h5ad              # GEARS processed data
  split: /path/adamson/splits/adamson_simulation_1_0.75.pkl
  split_name: simulation
  split_seed: 1
  train_gene_set_size: 0.75
  gears_root: /path            # GEARS loads <gears_root>/<gears_name>/
  gears_name: adamson
```

`split` must be the split pickle that GEARS produces for `split_name`,
`split_seed` and `train_gene_set_size`. The report uses it for the
train-mean-Δ baseline, so it must match what the model was trained on.

## Adding a dataset

1. Put the data where the pipeline machine can read it. For PP, run GEARS
   once so that `data_pyg/` and `splits/` exist.
2. Copy an existing config, then edit `name`, `label` and the paths.
3. Check it: `deepsc list`, then a quick smoke run, e.g.
   `deepsc finetune --task cta --dataset <name> --model deepsc_pretrain_1215 --epochs 1 --limit-steps 20 --run-id smoke`.
