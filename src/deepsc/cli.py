"""``deepsc`` command line: one entry point for the downstream pipeline.

    deepsc infer          run a model on a benchmark dataset -> prediction set
    deepsc report         prediction sets -> figures, tables, captions; model comparison
    deepsc import-legacy  convert results of the 2026-11-20 external runs
    deepsc list           show registered datasets and models

Equivalent: ``python -m deepsc <command> ...``; each command is also
``python -m deepsc.pipeline.<module>``.
"""

from __future__ import annotations

import argparse

COMMANDS = {
    "infer": (
        "deepsc.pipeline.infer",
        "run a model on a benchmark dataset and write a prediction set",
    ),
    "report": (
        "deepsc.pipeline.report",
        "prediction sets -> figures, tables and captions, per model and compared",
    ),
    "import-legacy": (
        "deepsc.pipeline.legacy",
        "convert test_predictions.npz of the 2026-11-20 runs",
    ),
}


def _list(_args) -> None:
    from deepsc.pipeline.registry import list_datasets, list_models

    for task, names in list_datasets().items():
        print(f"datasets/{task}: {', '.join(names) or '-'}")
    print(f"models: {', '.join(list_models()) or '-'}")


def main(argv=None) -> None:
    import importlib

    parser = argparse.ArgumentParser(
        prog="deepsc", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (module, help_) in COMMANDS.items():
        mod = importlib.import_module(module)
        sp = sub.add_parser(name, help=help_, description=mod.__doc__)
        mod.add_arguments(sp)
        sp.set_defaults(handler=mod.main_from_args)
    sub.add_parser("list", help="show registered datasets and models").set_defaults(
        handler=_list
    )
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
