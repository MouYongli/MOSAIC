"""Helpers shared by the task-specific figure modules."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Results of the 2026-11-20 downstream validation run on server warhol.
DOWNSTREAM_RUN_ROOT = Path("/DATA2/DeepSC/results/downstream_20261120")


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


def savefig(fig, path: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def use_headless_backend() -> None:
    """Select the non-interactive backend; call from CLI entry points only."""
    import matplotlib

    matplotlib.use("Agg")
