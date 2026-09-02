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

    snapshots = _normalize_snapshots(
        backend.get("snapshots", []),
        problem["qubo"],
        expected_count=normalized.get("num_snapshots", 0),
    )
    if snapshots:
        final_snapshot = snapshots[-1]
        if final_snapshot["solution"] is None:
            raise RuntimeError("The final solver snapshot does not contain an incumbent")
        snapshot_solution = final_snapshot["solution"]
        if snapshot_solution["sample"] != sample or not math.isclose(
            snapshot_solution["energy"], recomputed, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError("The final solver snapshot does not match the final solution")

    identity_parameters = dict(normalized)
    # Preserve run IDs produced before snapshot support when snapshots are disabled.
    if identity_parameters.get("num_snapshots") == 0:
        identity_parameters.pop("num_snapshots")
    identity = {
        "instance_id": problem["instance_id"],
        "solver_type": solver_type,
        "parameters": identity_parameters,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    result = {
        "schema_version": 2,
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
        "snapshots": snapshots,
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
        "snapshots": len(result.get("snapshots", [])),
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


def _normalize_snapshots(snapshots, qubo: dict, expected_count: int) -> list[dict]:
    if not isinstance(snapshots, list):
        raise RuntimeError("Solver snapshots must be returned as a list")
    if len(snapshots) != expected_count:
        raise RuntimeError(
            f"Solver returned {len(snapshots)} snapshots; expected {expected_count}"
        )

    normalized = []
    previous_progress = 0.0
    for expected_index, snapshot in enumerate(snapshots, start=1):
        if not isinstance(snapshot, dict):
            raise RuntimeError("Every solver snapshot must be a mapping")
        if snapshot.get("index") != expected_index:
            raise RuntimeError("Solver snapshot indices must be consecutive and one-based")

        progress = snapshot.get("progress_fraction")
        if not isinstance(progress, (int, float)) or isinstance(progress, bool):
            raise RuntimeError("Snapshot progress_fraction must be numeric")
        progress = float(progress)
        expected_progress = expected_index / expected_count
        if not math.isfinite(progress) or not math.isclose(
            progress, expected_progress, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError("Solver snapshots must be equally spaced through progress 1.0")
        if progress <= previous_progress:
            raise RuntimeError("Solver snapshot progress must be strictly increasing")
        previous_progress = progress

        target = snapshot.get("target")
        if not isinstance(target, dict):
            raise RuntimeError("Every solver snapshot must describe its work target")
        unit = target.get("unit")
        value = target.get("value")
        total = target.get("total")
        if not isinstance(unit, str) or not unit:
            raise RuntimeError("Snapshot target unit must be a non-empty string")
        for name, number in (("value", value), ("total", total)):
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(float(number))
                or number < 0
            ):
                raise RuntimeError(f"Snapshot target {name} must be a finite non-negative number")
        if value > total:
            raise RuntimeError("Snapshot target value cannot exceed its total")

        elapsed = snapshot.get("elapsed_seconds")
        if elapsed is not None and (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise RuntimeError("Snapshot elapsed_seconds must be finite and non-negative")

        output = {
            "index": expected_index,
            "progress_fraction": progress,
            "target": {"unit": unit, "value": value, "total": total},
            "elapsed_seconds": float(elapsed) if elapsed is not None else None,
            "solution": None,
            "verification": None,
            "solver_metrics": snapshot.get("metrics", {}),
        }
        snapshot_sample = snapshot.get("sample")
        reported_energy = snapshot.get("reported_energy")
        if snapshot_sample is not None:
            values = [int(item) for item in snapshot_sample]
            if len(values) != int(qubo["num_variables"]):
                raise RuntimeError(
                    f"Snapshot {expected_index} returned {len(values)} values for a "
                    f"{qubo['num_variables']}-variable QUBO"
                )
            if any(item not in (0, 1) for item in values):
                raise RuntimeError(f"Snapshot {expected_index} returned a non-binary sample")
            recomputed = qubo_energy(qubo, values)
            reported = float(reported_energy)
            difference = abs(recomputed - reported)
            tolerance = 1e-5 * max(1.0, abs(recomputed), abs(reported))
            if not math.isfinite(reported) or difference > tolerance:
                raise RuntimeError(
                    f"Snapshot {expected_index} reported energy {reported}, but independent "
                    f"QUBO evaluation gave {recomputed}"
                )
            output["solution"] = {"sample": values, "energy": recomputed}
            output["verification"] = {
                "binary_sample": True,
                "reported_energy": reported,
                "recomputed_energy": recomputed,
                "absolute_energy_difference": difference,
                "passed": True,
            }
        elif reported_energy is not None:
            raise RuntimeError("A snapshot without a sample cannot report an energy")
        normalized.append(output)
    return normalized


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
