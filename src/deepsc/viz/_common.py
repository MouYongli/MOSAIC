"""Helpers shared by the task-specific figure modules.

Output layout written by :class:`FigureWriter`::

    <out>/<task>/<dataset>/
        README.md        index: every figure with its title and short caption
        manifest.json    run record: inputs, verification results, code version, figure list
        figures/<task>_<dataset>_<name>.{pdf,png,json}
        tables/<task>_<dataset>_<name>.csv

Figures carry no titles. Each figure's ``.json`` sidecar holds ``title``,
``caption_short``, ``caption_long`` and ``description`` (plain-language reading
guide with the key numbers), plus ``metrics`` and ``sources``.
"""

from __future__ import annotations
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from datetime import datetime, timezone

# Results of the 2026-11-20 downstream validation run on server warhol.
DOWNSTREAM_RUN_ROOT = Path("/DATA2/DeepSC/results/downstream_20261120")
FORMATS = ("pdf", "png")
PNG_DPI = 300


class VerificationError(RuntimeError):
    """Raised when saved predictions do not match their source data or run record."""


@dataclass
class Checks:
    """Named pass/fail checks; ``require()`` raises if any failed."""

    results: dict[str, bool] = field(default_factory=dict)

    def add(self, name: str, ok) -> bool:
        self.results[name] = bool(ok)
        return bool(ok)

    def close(self, name: str, value: float, expected: float, atol: float) -> bool:
        return self.add(
            f"{name} {value:.6f} == recorded {expected:.6f}",
            np.isclose(value, expected, atol=atol),
        )

    def report(self, printer=print) -> None:
        for name, ok in self.results.items():
            printer(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    def require(self) -> None:
        failed = [name for name, ok in self.results.items() if not ok]
        if failed:
            raise VerificationError("Verification failed: " + "; ".join(failed))


def read_status(run_dir: Path) -> dict:
    with open(Path(run_dir) / "status.json") as f:
        return json.load(f)


def spread_labels(y: np.ndarray, gap: float) -> np.ndarray:
    """Return ascending label positions pushed apart so neighbours are >= ``gap`` apart."""
    y = np.asarray(y, dtype=float).copy()
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    return y


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def use_headless_backend() -> None:
    """Select the non-interactive backend; call from CLI entry points only."""
    import matplotlib

    matplotlib.use("Agg")


def apply_style() -> None:
    """Publication defaults: editable (Type 42) text in vector output, no figure titles."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _git_version() -> str | None:
    try:
        root = Path(__file__).resolve().parent
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class FigureRecord:
    id: str
    title: str
    caption_short: str
    caption_long: str
    description: str
    files: dict[str, str]
    metrics: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


class FigureWriter:
    """Saves untitled figures with a metadata sidecar and writes the dataset index."""

    def __init__(
        self,
        out: Path,
        task: str,
        dataset: str,
        dataset_label: str,
        sources: dict[str, Path],
        formats=FORMATS,
    ):
        self.task, self.dataset, self.dataset_label = task, slug(dataset), dataset_label
        self.root = Path(out) / task / self.dataset
        self.fig_dir, self.table_dir = self.root / "figures", self.root / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.sources = {k: str(v) for k, v in sources.items()}
        self.formats = formats
        self.records: list[FigureRecord] = []
        self.tables: list[str] = []

    def _id(self, name: str) -> str:
        return f"{self.task}_{self.dataset}_{name}"

    def save(
        self,
        fig,
        name: str,
        *,
        title: str,
        caption_short: str,
        caption_long: str,
        description: str,
        metrics: dict | None = None,
        sources: list[str] | None = None,
    ) -> FigureRecord:
        import matplotlib.pyplot as plt

        fid = self._id(name)
        files = {}
        for fmt in self.formats:
            path = self.fig_dir / f"{fid}.{fmt}"
            fig.savefig(path, dpi=PNG_DPI if fmt == "png" else None)
            files[fmt] = str(path.relative_to(self.root))
        plt.close(fig)
        rec = FigureRecord(
            fid,
            title,
            caption_short,
            caption_long,
            description,
            files,
            metrics or {},
            sources or list(self.sources),
        )
        meta = {"task": self.task, "dataset": self.dataset, **asdict(rec)}
        (self.fig_dir / f"{fid}.json").write_text(
            json.dumps(meta, indent=2, default=_json_default) + "\n"
        )
        self.records.append(rec)
        return rec

    def table(self, df, name: str, **to_csv) -> Path:
        path = self.table_dir / f"{self._id(name)}.csv"
        df.to_csv(path, **to_csv)
        self.tables.append(str(path.relative_to(self.root)))
        return path

    def finalize(self, checks: Checks, summary: dict | None = None) -> Path:
        manifest = {
            "task": self.task,
            "dataset": self.dataset,
            "dataset_label": self.dataset_label,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": _git_version(),
            "sources": self.sources,
            "verification": checks.results,
            "summary": summary or {},
            "figures": [
                {
                    "id": r.id,
                    "title": r.title,
                    "files": r.files,
                    "metadata": f"figures/{r.id}.json",
                }
                for r in self.records
            ],
            "tables": self.tables,
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default) + "\n"
        )

        lines = [
            f"# {self.task.upper()} · {self.dataset_label}",
            "",
            "Auto-generated by `deepsc.viz`. Figures have no titles; titles and captions are in each "
            "figure's `.json` file.",
            "",
            "## Verification",
            "",
        ]
        lines += [
            f"- {'PASS' if ok else 'FAIL'}: {name}"
            for name, ok in checks.results.items()
        ]
        for i, r in enumerate(self.records, 1):
            lines += [
                "",
                f"## {i}. {r.title}",
                "",
                f"`{r.id}`",
                "",
                f"![{r.title}]({r.files.get('png', '')})",
                "",
                f"**{r.caption_short}** {r.caption_long}",
                "",
                f"*Description.* {r.description}",
            ]
        if self.tables:
            lines += ["", "## Tables", ""] + [f"- `{t}`" for t in self.tables]
        readme = self.root / "README.md"
        readme.write_text("\n".join(lines) + "\n")
        return readme
