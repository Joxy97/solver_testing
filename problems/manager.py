"""Discover, validate, generate, save, and load problem instances."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from . import (
    graph_coloring,
    independent_set,
    max_cut,
    max_sat,
    maximum_clique,
    number_partitioning,
    quadratic_knapsack,
    random_qubo,
    set_packing,
    spin_glass,
)
from .validation import WARNING_CODES, validate_problem


PROBLEMS = {
    "max_cut": max_cut,
    "independent_set": independent_set,
    "maximum_clique": maximum_clique,
    "graph_coloring": graph_coloring,
    "number_partitioning": number_partitioning,
    "quadratic_knapsack": quadratic_knapsack,
    "set_packing": set_packing,
    "max_sat": max_sat,
    "spin_glass": spin_glass,
    "random_qubo": random_qubo,
}


class ProblemValidationError(ValueError):
    """Raised when a generated instance fails a declared validation gate."""

    def __init__(self, problem: dict, rejected_warnings: list[dict] | None = None):
        self.problem = problem
        self.rejected_warnings = rejected_warnings or []
        messages = [issue["message"] for issue in problem["validation"]["errors"]]
        messages.extend(issue["message"] for issue in self.rejected_warnings)
        super().__init__("Generated instance failed validation: " + " ".join(messages))


def list_problems() -> list[dict]:
    return [
        {
            "type": problem_type,
            "name": module.NAME,
            "description": module.DESCRIPTION.strip(),
        }
        for problem_type, module in PROBLEMS.items()
    ]


def describe_problem(problem_type: str) -> dict:
    module = _module(problem_type)
    return {
        "type": problem_type,
        "name": module.NAME,
        "description": module.DESCRIPTION.strip(),
        "parameters": copy.deepcopy(module.PARAMETERS),
    }


def generate_problem(
    problem_type: str,
    parameters: dict | None = None,
    seed: int = 0,
    reject_warnings: Iterable[str] | None = None,
    progress=None,
) -> dict:
    module = _module(problem_type)
    normalized = _validate_parameters(module.PARAMETERS, parameters or {})
    if problem_type == "random_qubo":
        generated = module.generate(normalized, int(seed), progress=progress)
    else:
        generated = module.generate(normalized, int(seed))
    resolved_parameters = generated.pop("parameters", normalized)
    problem = {
        "schema_version": 2,
        "problem_type": problem_type,
        "problem_name": module.NAME,
        "seed": int(seed),
        "parameters": resolved_parameters,
        **generated,
    }
    problem["validation"] = revalidate_problem(problem)
    rejected_codes = set(reject_warnings or [])
    unknown_warning_codes = sorted(rejected_codes.difference(WARNING_CODES))
    if unknown_warning_codes:
        raise ValueError(
            "Unknown validation warning code(s): " + ", ".join(unknown_warning_codes)
        )
    rejected = [
        warning
        for warning in problem["validation"]["warnings"]
        if warning["code"] in rejected_codes
    ]
    if problem["validation"]["errors"] or rejected:
        raise ProblemValidationError(problem, rejected)
    problem["instance_id"] = _instance_identifier(problem)
    return problem


def revalidate_problem(problem: dict) -> dict:
    """Re-run deterministic validation on a generated or loaded problem instance."""
    if "problem_type" not in problem:
        raise ValueError("Problem is missing problem_type")
    module = _module(problem["problem_type"])
    return validate_problem(problem, module.validate)


def save_problem(problem: dict, destination: str | Path) -> Path:
    problem["schema_version"] = 2
    problem["validation"] = revalidate_problem(problem)
    if problem["validation"]["errors"]:
        raise ProblemValidationError(problem)
    problem["instance_id"] = _instance_identifier(problem)
    path = Path(destination)
    if path.suffix.lower() != ".json":
        path = path / f"{problem['instance_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(problem, handle, indent=2, sort_keys=False)
        handle.write("\n")
    temporary.replace(path)
    _update_manifest(problem, path)
    return path


def load_problem(source: str | Path) -> dict:
    path = Path(source)
    with path.open("r", encoding="utf-8") as handle:
        problem = json.load(handle)
    required = {"schema_version", "problem_type", "instance_id", "qubo", "problem_data"}
    missing = sorted(required.difference(problem))
    if missing:
        raise ValueError(f"Not a valid saved problem; missing: {', '.join(missing)}")
    if problem.get("schema_version", 1) >= 2 and "validation" not in problem:
        raise ValueError("Not a valid schema-version-2 problem; missing validation metadata")
    return problem


def _update_manifest(problem: dict, problem_path: Path) -> None:
    manifest_path = problem_path.parent / "manifest.csv"
    existing_rows = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    row = _manifest_row(problem, problem_path)
    rows_by_id = {
        existing["instance_id"]: existing
        for existing in existing_rows
        if existing.get("instance_id")
    }
    rows_by_id[problem["instance_id"]] = row
    base_fields = [
        "instance_id",
        "problem_type",
        "problem_name",
        "seed",
        "file",
        "num_variables",
        "linear_terms",
        "quadratic_terms",
        "validation_status",
        "warning_codes",
        "error_codes",
    ]
    extra_fields = sorted(
        {
            key
            for existing in rows_by_id.values()
            for key in existing
            if key not in base_fields
        }
    )
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        for existing in sorted(rows_by_id.values(), key=lambda item: item["instance_id"]):
            writer.writerow(existing)
    temporary.replace(manifest_path)


def _manifest_row(problem: dict, problem_path: Path) -> dict:
    qubo = problem["qubo"]
    validation = problem["validation"]
    row = {
        "instance_id": problem["instance_id"],
        "problem_type": problem["problem_type"],
        "problem_name": problem["problem_name"],
        "seed": problem["seed"],
        "file": problem_path.name,
        "num_variables": qubo["num_variables"],
        "linear_terms": len(qubo["linear"]),
        "quadratic_terms": len(qubo["quadratic"]),
        "validation_status": validation["status"],
        "warning_codes": ";".join(issue["code"] for issue in validation["warnings"]),
        "error_codes": ";".join(issue["code"] for issue in validation["errors"]),
    }
    row.update(_flatten_for_manifest("parameter", problem["parameters"]))
    row.update(_flatten_for_manifest("stat", validation["characteristics"]))
    return row


def _flatten_for_manifest(prefix: str, value) -> dict:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            flattened.update(_flatten_for_manifest(f"{prefix}.{key}", child))
        return flattened
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value, separators=(",", ":"))}
    return {prefix: value}


def _module(problem_type: str):
    try:
        return PROBLEMS[problem_type]
    except KeyError as error:
        available = ", ".join(PROBLEMS)
        raise ValueError(f"Unknown problem type '{problem_type}'. Available: {available}") from error


def _instance_identifier(problem: dict) -> str:
    if problem.get("problem_type") == "random_qubo":
        return f"seed_{int(problem['seed'])}"
    content = {key: value for key, value in problem.items() if key != "instance_id"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{problem['problem_type']}-{digest[:16]}"


def _validate_parameters(schema: dict, supplied: dict) -> dict:
    unknown = sorted(set(supplied).difference(schema))
    if unknown:
        raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")

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
            raise ValueError(f"Parameter '{name}' must be {expected_text}; received {value!r}")
        if "min" in specification and value < specification["min"]:
            raise ValueError(f"Parameter '{name}' must be at least {specification['min']}")
        if "max" in specification and value > specification["max"]:
            raise ValueError(f"Parameter '{name}' must be at most {specification['max']}")
        normalized[name] = value
    return normalized
