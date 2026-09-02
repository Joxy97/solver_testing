"""Friendly command-line entry point for solving saved QUBO problems."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import yaml

from problems.manager import load_problem
from solvers.manager import (
    SolverCapabilityError,
    describe_solver,
    list_solvers,
    save_result,
    solve_problem,
)


def _format_default(value) -> str:
    if value is None:
        return "automatic"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _print_solver_list() -> None:
    print("Available QUBO solvers:\n")
    for index, solver in enumerate(list_solvers(), start=1):
        print(f"{index:>2}. {solver['type']:<24} {solver['name']}")
        print(f"    {solver['description']}")


def _print_description(solver_type: str) -> None:
    description = describe_solver(solver_type)
    print(f"{description['name']} ({solver_type})")
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


def _solve_and_save(
    problem_path: Path,
    solver_type: str,
    parameters: dict,
    output: Path,
) -> Path:
    problem = load_problem(problem_path)
    result = solve_problem(problem, solver_type, parameters, source=problem_path)
    path = save_result(result, output)
    print(
        f"{problem['instance_id']} with {solver_type}: {result['status']}, "
        f"energy={result['solution']['energy']:.12g}, "
        f"time={result['timing']['wall_seconds']:.3f}s -> {path}"
    )
    return path


def _expand_problem_files(patterns: list[str], config_path: Path) -> list[Path]:
    files: set[Path] = set()
    for text in patterns:
        if not isinstance(text, str):
            raise ValueError("Every problem_files entry must be a path string")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        if candidate.is_dir():
            matches = candidate.rglob("*.json")
        elif any(character in str(candidate) for character in "*?["):
            matches = (Path(match) for match in glob.glob(str(candidate), recursive=True))
        else:
            matches = [candidate]
        files.update(path.resolve() for path in matches if path.is_file())
    if not files:
        raise ValueError("problem_files did not match any JSON problem files")
    return sorted(files)


def _run_config(path: Path) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("The solver configuration file must contain a YAML mapping")

    patterns = config.get("problem_files")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("Configuration must contain a non-empty problem_files list")
    problem_files = _expand_problem_files(patterns, path)
    problem_limit = config.get("problem_limit")
    if problem_limit is not None:
        if not isinstance(problem_limit, int) or isinstance(problem_limit, bool) or problem_limit < 1:
            raise ValueError("problem_limit must be a positive integer or null")
        problem_files = problem_files[:problem_limit]

    solver_entries = config.get("solvers")
    if not isinstance(solver_entries, list) or not solver_entries:
        raise ValueError("Configuration must contain a non-empty solvers list")

    output = Path(config.get("output_folder", "solver_results"))
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    continue_on_error = config.get("continue_on_error", False)
    if not isinstance(continue_on_error, bool):
        raise ValueError("continue_on_error must be true or false")

    saved = []
    failures = 0
    for problem_path in problem_files:
        for entry in solver_entries:
            if not isinstance(entry, dict) or "type" not in entry:
                raise ValueError("Every solvers entry must contain a type")
            parameters = entry.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError("Solver parameters must be a YAML mapping")
            try:
                saved.append(
                    _solve_and_save(problem_path, entry["type"], parameters, output)
                )
            except (ValueError, RuntimeError, OSError) as error:
                failures += 1
                if not continue_on_error:
                    raise
                print(
                    f"Skipped {problem_path.name} with {entry['type']}: {error}",
                    file=sys.stderr,
                )
    print(f"\nSaved {len(saved)} result(s) in {output}; {failures} failed or skipped.")
    return saved


def _prompt_value(name: str, specification: dict):
    default = specification.get("default")
    choices = specification.get("choices")
    print(f"\n{name}: {specification['description']}")
    if choices:
        print(f"Choices: {', '.join(map(str, choices))}")
    text = input(f"Value [{_format_default(default)}]: ").strip()
    return default if not text else yaml.safe_load(text)


def _guided_mode() -> None:
    print("QUBO Solver\n")
    problem_text = input("Path to a saved problem JSON: ").strip()
    if not problem_text:
        raise ValueError("A problem JSON path is required")
    problem_path = Path(problem_text)

    available = list_solvers()
    print()
    for index, solver in enumerate(available, start=1):
        print(f"{index:>2}. {solver['name']}")
    selection = input("\nSelect a solver number: ").strip()
    try:
        selected = available[int(selection) - 1]
    except (ValueError, IndexError) as error:
        raise ValueError("Please select one of the displayed solver numbers") from error

    description = describe_solver(selected["type"])
    print(f"\n{description['name']}: {description['description']}")
    print("Press Enter to accept each default.")
    parameters = {
        name: _prompt_value(name, specification)
        for name, specification in description["parameters"].items()
    }
    output_text = input("\nOutput folder [solver_results]: ").strip()
    _solve_and_save(
        problem_path,
        selected["type"],
        parameters,
        Path(output_text or "solver_results"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve saved QUBO instances and write verified JSON results."
    )
    parser.add_argument("config", nargs="?", type=Path, help="YAML solver configuration")
    parser.add_argument("--list", action="store_true", help="List available solvers")
    parser.add_argument("--describe", metavar="TYPE", help="Explain a solver and its parameters")
    parser.add_argument("--problem", type=Path, help="Saved QUBO problem JSON")
    parser.add_argument("--solver", metavar="TYPE", help="Solver to use")
    parser.add_argument(
        "--set",
        dest="assignments",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set a solver parameter; may be repeated",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("solver_results"), help="Result folder"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.list:
            _print_solver_list()
        elif args.describe:
            _print_description(args.describe)
        elif args.config:
            _run_config(args.config)
        elif args.problem or args.solver:
            if args.problem is None or args.solver is None:
                raise ValueError("--problem and --solver must be used together")
            _solve_and_save(
                args.problem,
                args.solver,
                _parse_assignments(args.assignments),
                args.output,
            )
        else:
            _guided_mode()
        return 0
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
