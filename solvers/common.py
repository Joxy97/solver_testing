"""Small helpers shared by the solver adapters."""

from __future__ import annotations

import importlib.metadata


class SolverCapabilityError(ValueError):
    """Raised when a valid QUBO is too large or unsuitable for a solver."""


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
