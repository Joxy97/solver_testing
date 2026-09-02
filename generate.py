"""Friendly command-line entry point for generating and saving QUBO problems."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from problems.manager import (
    ProblemValidationError,
    describe_problem,
    generate_problem,
    list_problems,
    save_problem,
)


def _print_problem_list() -> None:
    print("Available QUBO problem types:\n")
    for index, problem in enumerate(list_problems(), start=1):
        print(f"{index:>2}. {problem['type']:<22} {problem['name']}")
        print(f"    {problem['description']}")


def _format_default(value) -> str:
    if value is None:
        return "automatic"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _print_description(problem_type: str) -> None:
    description = describe_problem(problem_type)
    print(f"{description['name']} ({problem_type})")
    print(description["description"])
    print("\nParameters:\n")
    for name, specification in description["parameters"].items():
        default = _format_default(specification.get("default"))
        choices = specification.get("choices")
        suffix = f" Choices: {', '.join(map(str, choices))}." if choices else ""
        print(f"  {name}")
        print(f"    {specification['description']} Default: {default}.{suffix}")


def _parse_assignments(assignments: list[str]) -> dict:
    parameters = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected parameter=value, received: {assignment}")
        name, text = assignment.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing parameter name in: {assignment}")
        parameters[name] = yaml.safe_load(text)
    return parameters


def _save_generated(
    problem_type: str,
    parameters: dict,
    seed: int,
    output: Path,
    reject_warnings: list[str] | None = None,
    max_attempts: int = 1,
) -> Path:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    last_error = None
    problem = None
    for attempt in range(max_attempts):
        candidate_seed = seed + attempt * 1_000_000_000
        try:
            problem = generate_problem(
                problem_type,
                parameters,
                candidate_seed,
                reject_warnings=reject_warnings,
            )
            break
        except ProblemValidationError as error:
            last_error = error
            if attempt + 1 < max_attempts:
                print(
                    f"Rejected {problem_type} seed {candidate_seed}; "
                    "trying another deterministic seed."
                )
    if problem is None:
        raise last_error
    path = save_problem(problem, output)
    qubo = problem["qubo"]
    terms = len(qubo["linear"]) + len(qubo["quadratic"])
    print(
        f"Saved {problem['instance_id']}: "
        f"{qubo['num_variables']} variables, {terms} nonzero terms -> {path}"
    )
    warning_codes = [issue["code"] for issue in problem["validation"]["warnings"]]
    if warning_codes:
        print(f"  Validation warnings: {', '.join(warning_codes)}")
    return path


def _run_config(path: Path) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping")

    output = Path(config.get("output_folder", "generated_problems"))
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    base_seed = int(config.get("base_seed", 0))
    validation_config = config.get("validation", {})
    if not isinstance(validation_config, dict):
        raise ValueError("'validation' must be a YAML mapping")
    default_rejected = validation_config.get("reject_warnings", [])
    default_attempts = int(validation_config.get("max_attempts", 1))
    if not isinstance(default_rejected, list) or not all(
        isinstance(code, str) for code in default_rejected
    ):
        raise ValueError("validation.reject_warnings must be a list of warning-code strings")
    entries = config.get("problems")
    if entries is None and "problem" in config:
        entries = [config["problem"]]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Configuration must contain a non-empty 'problems' list")

    paths = []
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError("Every problem entry must be a mapping containing 'type'")
        count = int(entry.get("count", 1))
        if count < 1:
            raise ValueError("Problem count must be at least 1")
        first_seed = int(entry.get("seed", base_seed + entry_index * 1_000_000))
        parameters = entry.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Problem parameters must be a YAML mapping")
        entry_validation = entry.get("validation", {})
        if not isinstance(entry_validation, dict):
            raise ValueError("Problem-entry validation must be a YAML mapping")
        rejected_warnings = entry_validation.get("reject_warnings", default_rejected)
        max_attempts = int(entry_validation.get("max_attempts", default_attempts))
        if not isinstance(rejected_warnings, list) or not all(
            isinstance(code, str) for code in rejected_warnings
        ):
            raise ValueError("reject_warnings must be a list of warning-code strings")
        for instance_index in range(count):
            paths.append(
                _save_generated(
                    entry["type"],
                    parameters,
                    first_seed + instance_index,
                    output,
                    rejected_warnings,
                    max_attempts,
                )
            )
    print(f"\nGenerated {len(paths)} problem instance(s) in {output}")
    return paths


def _prompt_value(name: str, specification: dict):
    default = specification.get("default")
    choices = specification.get("choices")
    print(f"\n{name}: {specification['description']}")
    if choices:
        print(f"Choices: {', '.join(map(str, choices))}")
    text = input(f"Value [{_format_default(default)}]: ").strip()
    return default if not text else yaml.safe_load(text)


def _guided_mode() -> None:
    problems = list_problems()
    print("QUBO Problem Generator\n")
    for index, problem in enumerate(problems, start=1):
        print(f"{index:>2}. {problem['name']}")
    selection = input("\nSelect a problem number: ").strip()
    try:
        selected = problems[int(selection) - 1]
    except (ValueError, IndexError) as error:
        raise ValueError("Please select one of the displayed problem numbers") from error

    description = describe_problem(selected["type"])
    print(f"\n{description['name']}: {description['description']}")
    print("Press Enter to accept each default.")
    parameters = {
        name: _prompt_value(name, specification)
        for name, specification in description["parameters"].items()
    }
    count_text = input("\nNumber of instances [1]: ").strip()
    count = int(count_text) if count_text else 1
    if count < 1:
        raise ValueError("Number of instances must be at least 1")
    seed_text = input("Starting random seed [0]: ").strip()
    seed = int(seed_text) if seed_text else 0
    output_text = input("Output folder [generated_problems]: ").strip()
    output = Path(output_text or "generated_problems")
    print()
    for index in range(count):
        _save_generated(selected["type"], parameters, seed + index, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible QUBO problem instances and save them as JSON."
    )
    parser.add_argument("config", nargs="?", type=Path, help="YAML configuration file")
    parser.add_argument("--list", action="store_true", help="List available problems")
    parser.add_argument("--describe", metavar="TYPE", help="Explain a problem and its parameters")
    parser.add_argument("--problem", metavar="TYPE", help="Generate one problem type directly")
    parser.add_argument(
        "--set",
        dest="assignments",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set a problem parameter; may be repeated",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of instances")
    parser.add_argument("--seed", type=int, default=0, help="Starting random seed")
    parser.add_argument(
        "--output", type=Path, default=Path("generated_problems"), help="Output folder"
    )
    parser.add_argument(
        "--reject-warning",
        action="append",
        default=[],
        metavar="CODE",
        help="Reject and regenerate instances carrying this validation warning",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum deterministic generation attempts after validation rejection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.list:
            _print_problem_list()
        elif args.describe:
            _print_description(args.describe)
        elif args.config:
            _run_config(args.config)
        elif args.problem:
            if args.count < 1:
                raise ValueError("--count must be at least 1")
            parameters = _parse_assignments(args.assignments)
            for index in range(args.count):
                _save_generated(
                    args.problem,
                    parameters,
                    args.seed + index,
                    args.output,
                    args.reject_warning,
                    args.max_attempts,
                )
        else:
            _guided_mode()
        return 0
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
