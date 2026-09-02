"""Small helpers shared by the solver adapters."""

from __future__ import annotations

import importlib.metadata
import math


class SolverCapabilityError(ValueError):
    """Raised when a valid QUBO is too large or unsuitable for a solver."""


def equally_spaced_targets(total_work: int, num_snapshots: int) -> list[int]:
    """Return distinct integer checkpoints at equal fractions of a work budget."""
    if total_work < 1:
        raise ValueError("total_work must be at least 1")
    if num_snapshots < 0:
        raise ValueError("num_snapshots cannot be negative")
    if num_snapshots > total_work:
        raise ValueError(
            f"num_snapshots={num_snapshots} exceeds the {total_work} available work units"
        )
    if not num_snapshots:
        return []
    return [
        math.ceil(index * total_work / num_snapshots)
        for index in range(1, num_snapshots + 1)
    ]


def snapshot_record(
    *,
    index: int,
    count: int,
    unit: str,
    target: int | float,
    total: int | float,
    sample=None,
    reported_energy: float | None = None,
    elapsed_seconds: float | None = None,
    metrics: dict | None = None,
) -> dict:
    """Build the adapter-level representation of one optimization checkpoint."""
    record = {
        "index": index,
        "progress_fraction": index / count,
        "target": {"unit": unit, "value": target, "total": total},
        "sample": sample,
        "reported_energy": reported_energy,
        "metrics": metrics or {},
    }
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = elapsed_seconds
    return record


def package_version(distribution: str) -> str:
    """Return an installed package version without importing optional packages."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def require_package(import_name: str, distribution: str, solver_name: str) -> None:
    """Raise an actionable error for a missing optional solver dependency."""
    try:
        __import__(import_name)
    except ImportError as error:
        raise RuntimeError(
            f"{solver_name} requires '{distribution}'. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error
