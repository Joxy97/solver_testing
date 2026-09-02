"""Discover solver adapters, validate parameters, run them, and save results."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

from problems.common import qubo_energy

from . import exact, ocean_sa, scipy_highs, simulated_bifurcation
from .common import SolverCapabilityError


SOLVERS = {
    "exact": exact,
    "scipy_highs": scipy_highs,
    "ocean_sa": ocean_sa,
    "simulated_bifurcation": simulated_bifurcation,
}


def list_solvers() -> list[dict]:
    return [
        {
            "type": solver_type,
            "name": module.NAME,
            "description": module.DESCRIPTION.strip(),
        }
        for solver_type, module in SOLVERS.items()
    ]


def describe_solver(solver_type: str) -> dict:
    module = _module(solver_type)
    return {
        "type": solver_type,
        "name": module.NAME,
        "description": module.DESCRIPTION.strip(),
        "parameters": copy.deepcopy(module.PARAMETERS),
    }


def solve_problem(
    problem: dict,
    solver_type: str,
    parameters: dict | None = None,
    source: str | Path | None = None,
) -> dict:
    """Solve one loaded problem and return the normalized, verified result record."""
    module = _module(solver_type)
    normalized = _validate_parameters(module.PARAMETERS, parameters or {})
    started = time.perf_counter()
    backend = module.solve(problem["qubo"], normalized)
    wall_seconds = time.perf_counter() - started

    sample = [int(value) for value in backend["sample"]]
    if len(sample) != int(problem["qubo"]["num_variables"]):
        raise RuntimeError(
            f"{module.NAME} returned {len(sample)} values for a "
            f"{problem['qubo']['num_variables']}-variable QUBO"
        )
    if any(value not in (0, 1) for value in sample):
        raise RuntimeError(f"{module.NAME} returned a non-binary sample")

    recomputed = qubo_energy(problem["qubo"], sample)
    reported = float(backend.get("reported_energy", recomputed))
    difference = abs(recomputed - reported)
    tolerance = 1e-5 * max(1.0, abs(recomputed), abs(reported))
    if not math.isfinite(recomputed) or not math.isfinite(reported):
        raise RuntimeError(f"{module.NAME} returned a non-finite objective value")
    if difference > tolerance:
        raise RuntimeError(
            f"{module.NAME} reported energy {reported}, but independent QUBO evaluation "
            f"gave {recomputed} (difference {difference})."
        )

    identity = {
        "instance_id": problem["instance_id"],
        "solver_type": solver_type,
        "parameters": normalized,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    result = {
        "schema_version": 1,
        "run_id": f"{solver_type}-{digest}",
        "problem": {
            "instance_id": problem["instance_id"],
            "problem_type": problem["problem_type"],
            "num_variables": int(problem["qubo"]["num_variables"]),
            "quadratic_terms": len(problem["qubo"]["quadratic"]),
            "source": str(source) if source is not None else None,
        },
        "solver": {
            "type": solver_type,
            "name": module.NAME,
            "version": module.version(),
            "parameters": normalized,
            "device": backend.get("device", "cpu"),
        },
        "status": backend["status"],
        "solution": {
            "sample": sample,
            "energy": recomputed,
        },
        "timing": {"wall_seconds": wall_seconds},
        "verification": {
            "binary_sample": True,
            "reported_energy": reported,
            "recomputed_energy": recomputed,
            "absolute_energy_difference": difference,
            "passed": True,
        },
        "solver_metrics": backend.get("metrics", {}),
    }
    return _json_safe(result)


def save_result(result: dict, destination: str | Path) -> Path:
    """Atomically save a result and update a flat manifest in the result root."""
    destination = Path(destination)
    if destination.suffix.lower() == ".json":
        path = destination
        root = destination.parent
    else:
        root = destination
        path = (
            root
            / result["solver"]["type"]
            / f"{result['problem']['instance_id']}--{result['run_id']}.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=False)
        handle.write("\n")
    temporary.replace(path)
    _update_manifest(result, path, root)
    return path


def _update_manifest(result: dict, result_path: Path, root: Path) -> None:
    manifest = root / "manifest.csv"
    rows = []
    if manifest.exists():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    try:
        relative_path = result_path.relative_to(root)
    except ValueError:
        relative_path = result_path
    row = {
        "run_id": result["run_id"],
        "instance_id": result["problem"]["instance_id"],
        "problem_type": result["problem"]["problem_type"],
        "num_variables": result["problem"]["num_variables"],
        "quadratic_terms": result["problem"]["quadratic_terms"],
        "solver_type": result["solver"]["type"],
        "solver_version": result["solver"]["version"],
        "device": result["solver"]["device"],
        "status": result["status"],
        "energy": result["solution"]["energy"],
        "wall_seconds": result["timing"]["wall_seconds"],
        "file": str(relative_path),
    }
    by_key = {
        (existing.get("run_id"), existing.get("instance_id")): existing for existing in rows
    }
    by_key[(row["run_id"], row["instance_id"])] = row
    fields = list(row)
    temporary = manifest.with_suffix(".csv.tmp")
    root.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for existing in sorted(
            by_key.values(), key=lambda item: (item.get("solver_type", ""), item.get("instance_id", ""))
        ):
            writer.writerow({field: existing.get(field, "") for field in fields})
    temporary.replace(manifest)


def _module(solver_type: str):
    try:
        return SOLVERS[solver_type]
    except KeyError as error:
        available = ", ".join(SOLVERS)
        raise ValueError(f"Unknown solver '{solver_type}'. Available: {available}") from error


def _validate_parameters(schema: dict, supplied: dict) -> dict:
    unknown = sorted(set(supplied).difference(schema))
    if unknown:
        raise ValueError(f"Unknown solver parameter(s): {', '.join(unknown)}")

    normalized = {}
    for name, specification in schema.items():
        value = supplied.get(name, copy.deepcopy(specification.get("default")))
        if value is None and specification.get("allow_none"):
            normalized[name] = None
            continue
        expected = specification["type"]
        valid = {
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "string": isinstance(value, str),
            "choice": value in specification.get("choices", []),
        }.get(expected, False)
        if not valid:
            expected_text = (
                f"one of {specification['choices']}" if expected == "choice" else expected
            )
            raise ValueError(
                f"Solver parameter '{name}' must be {expected_text}; received {value!r}"
            )
        if "min" in specification and value < specification["min"]:
            raise ValueError(
                f"Solver parameter '{name}' must be at least {specification['min']}"
            )
        if "max" in specification and value > specification["max"]:
            raise ValueError(
                f"Solver parameter '{name}' must be at most {specification['max']}"
            )
        normalized[name] = value
    return normalized


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)
