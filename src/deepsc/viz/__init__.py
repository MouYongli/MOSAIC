"""Result figures for DeepSC downstream tasks, built from saved test predictions.

Each task module verifies the saved predictions against the source dataset and
the run record before plotting, and refuses to plot if any check fails:

    python -m deepsc.viz.cta           --help   # cell type annotation
    python -m deepsc.viz.perturbation  --help   # perturbation prediction
"""
