# MOSAIC downstream validation report — 2026-11-20

Report date: **2026-11-20**.

This report collects completed CTA and perturbation validation results for the
legacy DeepSC models retained in MOSAIC. Both training and independent test
workflows completed successfully on server `warhol`.

The date above labels this report. The results below are attributed to their
recorded DeepSC checkout and validation script; it is not a claim that MOSAIC's
original launchers produced these metrics on that date. Existing artifact paths
retain their original names so the source logs and checkpoints remain traceable.

## Which code and checkpoint were used?

The runs were executed from the local checkout `/home/angli/DeepSC`.

Both tasks initialized from `pretraining_1215`:

```text
/DATA2/DeepSC/checkpoints/pretraining_1215/latest_checkpoint.ckpt
```

This is an existing pretraining checkpoint, not a new pretraining run. It uses
256-dimensional embeddings, 10 transformer layers, 35,210 genes and 5 expression
bins; its saved epoch field is 11. Its SHA-256 is:

```text
51122b16038ceaeb74531e7fbd86034e5a8ae2105e2710d00c31bc37e69bcacd
```

The historical path `/DATA2/DeepSC/results/pretraining_1215/latest_checkpoint.ckpt`
is a symlink to the file above. The mapping used by both tasks is now bundled
in this repository as [gene_map_tp10k.csv](../../../resources/gene_maps/gene_map_tp10k.csv),
with contents identical to the file used in the recorded runs. Its local path is:

`/home/angli/MOSAIC/resources/gene_maps/gene_map_tp10k.csv`.

## What was run and what were the results?

Environment: existing Python 3.11 `deepsc` Conda environment, PyTorch
`2.5.1+cu121`, physical GPU 3 (H100 PCIe, 80 GB), BF16 autocast and full-parameter
fine-tuning. Both tasks used seed 42, gradient clipping at 1.0, a maximum of
10 epochs and early stopping with patience 3. Both finished with exit code 0.

| Item | Cell type annotation | Perturbation prediction |
| --- | --- | --- |
| Dataset | Zheng, 11 cell types | Adamson, GEARS simulation split, seed 1 |
| Train / validation / test cells | 52,748 / 6,589 / 6,606 | 52,730 / 3,428 / 9,741 |
| Training batch size | 16 | 8; evaluation batch size 4 |
| Training sequence length | 1,024 | Up to 1,536 genes; full gene input for evaluation |
| Optimizer | Adam, constant learning rate `1e-4` | Adam, initial learning rate `1e-4`, multiplied by 0.9 each epoch |
| Best checkpoint selection | Minimum validation loss | Maximum validation Pearson delta |
| Best epoch / final epoch | 6 / 9 | 3 / 6 |

CTA reads `/DATA2/DeepSC/cell_type_annotation/zheng_{train,val,test}.h5ad` and
applies TP10K + log1p normalization. Its pretrained encoder loads strictly; the
classification head is newly initialized.

PP reads `/DATA2/DeepSC/adamson/perturb_processed.h5ad`, the existing
`data_pyg/cell_graphs.pkl`, and `splits/adamson_simulation_1_0.75.pkl` under that
dataset directory. Of 5,060 measured genes, 4,064 map to the vocabulary. The
continuous expression encoder and perturbation encoder are newly initialized;
compatible pretraining weights are loaded for the other components.

Final independent test results, using the best validation checkpoint:

| Metric | Value |
| --- | ---: |
| CTA accuracy | 82.41% |
| CTA macro-F1 | 0.702821 |
| CTA cross-entropy loss | 0.499014 |
| PP Pearson | 0.989069 |
| PP DE Pearson | 0.972033 |
| PP Pearson delta | 0.698090 |
| PP DE Pearson delta | 0.843787 |
| PP MSE | 0.130913 |
| PP control-input baseline MSE | 0.231165 |

PP MSE is calculated over mapped genes. For correlation metrics, unmapped
columns are zeroed in the arrays passed to the original DeepSC metric function.
Correlations use condition means; delta subtracts the control mean, and DE uses
the function's top 20 differentially expressed genes.

## Where are the weights and results?

All paths below are on server `warhol`; weights and large predictions remain
outside Git.

| Artifact | Path |
| --- | --- |
| Pretraining weights | `/DATA2/DeepSC/checkpoints/pretraining_1215/latest_checkpoint.ckpt` |
| CTA best and final checkpoints | `/DATA2/DeepSC/checkpoints/cta_zheng_1215/{best_model.pt,last.pt}` |
| PP best and final checkpoints | `/DATA2/DeepSC/checkpoints/pp_adamson_1215/{best_model.pt,last.pt}` |
| CTA results | `/DATA2/DeepSC/results/downstream_20261120/cta_zheng_1215/` |
| PP results | `/DATA2/DeepSC/results/downstream_20261120/pp_adamson_1215/` |
| CTA log | `/DATA2/DeepSC/results/downstream_20261120/cta.log` |
| PP log | `/DATA2/DeepSC/results/downstream_20261120/pp.log` |

Each result directory contains `status.json` (configuration, epoch history and
test metrics), `events.jsonl`, `exit_code` and `test_predictions.npz`. The original
working directory `/home/angli/deepsc_gpu_runs_20261120/` remains available. Original
launch records are also archived under
`/DATA2/DeepSC/results/downstream_20261120/recorded_run/`.
