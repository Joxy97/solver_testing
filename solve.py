"""Friendly command-line entry point for solving saved QUBO problems."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import multiprocessing
import os
import sys
from dataclasses import dataclass
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


@dataclass
class _SolveTask:
    problem_path: Path
    solver_type: str
    parameters: dict
    instance_name: str
    output_path: Path
    devices: tuple[str, ...]


def _solve_worker(
    problem_path: str, solver_type: str, parameters: dict, instance_name: str
) -> dict:
    """Process-pool entry point. The parent process alone writes result files."""
    source = Path(problem_path)
    problem = load_problem(source)
    return solve_problem(
        problem,
        solver_type,
        parameters,
        source=source,
        instance_name=instance_name,
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
    instance_name: str | None = None,
) -> Path:
    problem = load_problem(problem_path)
    result = solve_problem(
        problem,
        solver_type,
        parameters,
        source=problem_path,
        instance_name=instance_name,
    )
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


def _available_directory(path: Path) -> Path:
    if not path.exists():
        return path
    enumerator = 2
    while True:
        candidate = path.with_name(f"{path.name}({enumerator})")
        if not candidate.exists():
            return candidate
        enumerator += 1


def _run_names(entry: dict) -> list[str]:
    replicas = entry.get("replicas", 1)
    if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 1:
        raise ValueError("Solver replicas must be a positive integer")
    base = entry["instance_name"].strip()
    if replicas == 1:
        return [base]
    return [f"{base}_replica_{index}" for index in range(1, replicas + 1)]


def _entry_devices(entry: dict, solver_parameters: dict) -> tuple[str, ...]:
    devices = entry.get("devices")
    if devices is not None:
        if (
            not isinstance(devices, list)
            or not devices
            or not all(isinstance(device, str) and device.startswith("cuda:") for device in devices)
        ):
            raise ValueError("devices must be a non-empty list such as [cuda:0, cuda:1]")
        normalized = tuple(device.strip().lower() for device in devices)
        if len(normalized) != len(set(normalized)):
            raise ValueError("A solver's devices list cannot contain duplicates")
        if "device" not in describe_solver(entry["type"])["parameters"]:
            raise ValueError(f"Solver {entry['type']} does not support CUDA device selection")
        return normalized
    requested = solver_parameters.get("device", "cpu")
    if isinstance(requested, str) and requested.strip().lower().startswith("cuda"):
        value = requested.strip().lower()
        return ("cuda:0" if value == "cuda" else value,)
    return ()


def _save_completed(result: dict, path: Path) -> Path:
    saved = save_result(result, path)
    print(
        f"{result['problem']['instance_id']} with {result['solver']['instance_name']}: "
        f"{result['status']}, energy={result['solution']['energy']:.12g}, "
        f"time={result['timing']['wall_seconds']:.3f}s -> {saved}"
    )
    return saved


def _run_tasks_sequential(tasks: list[_SolveTask], continue_on_error: bool) -> tuple[list[Path], int]:
    saved = []
    failures = 0
    device_positions: dict[tuple[str, ...], int] = {}
    for index, task in enumerate(tasks, start=1):
        parameters = dict(task.parameters)
        if task.devices:
            position = device_positions.get(task.devices, 0)
            parameters["device"] = task.devices[position % len(task.devices)]
            device_positions[task.devices] = position + 1
        print(f"[{index}/{len(tasks)}] Starting {task.problem_path.name} x {task.instance_name}")
        try:
            result = _solve_worker(
                str(task.problem_path), task.solver_type, parameters, task.instance_name
            )
            saved.append(_save_completed(result, task.output_path))
        except (ValueError, RuntimeError, OSError) as error:
            failures += 1
            if not continue_on_error:
                raise
            print(f"Skipped {task.problem_path.name} with {task.instance_name}: {error}", file=sys.stderr)
    return saved, failures


def _run_tasks_parallel(
    tasks: list[_SolveTask], continue_on_error: bool, cpu_workers: int
) -> tuple[list[Path], int]:
    gpu_devices = sorted({device for task in tasks for device in task.devices})
    cpu_tasks_exist = any(not task.devices for task in tasks)
    process_context = multiprocessing.get_context("spawn")
    cpu_executor = (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=cpu_workers, mp_context=process_context
        )
        if cpu_tasks_exist
        else None
    )
    gpu_executors = {
        device: concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=process_context
        )
        for device in gpu_devices
    }
    future_tasks: dict[concurrent.futures.Future, tuple[_SolveTask, str]] = {}
    gpu_queue = [task for task in tasks if task.devices]
    free_gpus = set(gpu_devices)
    saved = []
    failures = 0
    try:
        for task in tasks:
            if task.devices:
                continue
            parameters = dict(task.parameters)
            future = cpu_executor.submit(
                _solve_worker,
                str(task.problem_path),
                task.solver_type,
                parameters,
                task.instance_name,
            )
            future_tasks[future] = (task, "cpu")
            print(f"Queued {task.problem_path.name} x {task.instance_name} on cpu")

        completed = 0
        while gpu_queue or future_tasks:
            for device in sorted(free_gpus):
                position = next(
                    (index for index, task in enumerate(gpu_queue) if device in task.devices),
                    None,
                )
                if position is None:
                    continue
                task = gpu_queue.pop(position)
                parameters = dict(task.parameters)
                parameters["device"] = device
                future = gpu_executors[device].submit(
                    _solve_worker,
                    str(task.problem_path),
                    task.solver_type,
                    parameters,
                    task.instance_name,
                )
                future_tasks[future] = (task, device)
                free_gpus.remove(device)
                print(f"Queued {task.problem_path.name} x {task.instance_name} on {device}")

            if not future_tasks:
                raise ValueError("No configured device can run the remaining GPU tasks")
            done, _ = concurrent.futures.wait(
                future_tasks, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                task, resource = future_tasks.pop(future)
                if resource != "cpu":
                    free_gpus.add(resource)
                completed += 1
                try:
                    saved.append(_save_completed(future.result(), task.output_path))
                except (ValueError, RuntimeError, OSError) as error:
                    failures += 1
                    if not continue_on_error:
                        for remaining in future_tasks:
                            remaining.cancel()
                        raise
                    print(
                        f"Skipped {task.problem_path.name} with {task.instance_name}: {error}",
                        file=sys.stderr,
                    )
                print(f"Progress: {completed}/{len(tasks)} tasks finished")
    finally:
        if cpu_executor is not None:
            cpu_executor.shutdown(cancel_futures=True)
        for executor in gpu_executors.values():
            executor.shutdown(cancel_futures=True)
    return saved, failures


def _run_config(path: Path) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("The solver configuration file must contain a YAML mapping")

    patterns = config.get("problem_files")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("Configuration must contain a non-empty problem_files list")
    problem_files = _expand_problem_files(patterns, path)
    problem_root_value = config.get("problem_root")
    if not isinstance(problem_root_value, str) or not problem_root_value.strip():
        raise ValueError("Configuration must contain a non-empty problem_root path")
    problem_root = Path(problem_root_value)
    if not problem_root.is_absolute():
        problem_root = (path.parent / problem_root).resolve()
    if not problem_root.is_dir():
        raise ValueError(f"problem_root is not a directory: {problem_root}")
    outside_root = [item for item in problem_files if not item.is_relative_to(problem_root)]
    if outside_root:
        raise ValueError(f"All problem_files must be inside problem_root: {outside_root[0]}")
    problem_limit = config.get("problem_limit")
    if problem_limit is not None:
        if not isinstance(problem_limit, int) or isinstance(problem_limit, bool) or problem_limit < 1:
            raise ValueError("problem_limit must be a positive integer or null")
        problem_files = problem_files[:problem_limit]

    solver_entries = config.get("solvers")
    if not isinstance(solver_entries, list) or not solver_entries:
        raise ValueError("Configuration must contain a non-empty solvers list")
    supplied_names = []
    all_run_names = []
    for entry in solver_entries:
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError("Every solvers entry must contain a type")
        if not isinstance(entry.get("parameters", {}), dict):
            raise ValueError("Solver parameters must be a YAML mapping")
        instance_name = entry.get("instance_name")
        if not isinstance(instance_name, str) or not instance_name.strip():
            raise ValueError("Every solvers entry must contain a non-empty instance_name")
        instance_name = instance_name.strip()
        if Path(instance_name).name != instance_name or any(
            character in instance_name for character in '<>:"/\\|?*'
        ):
            raise ValueError("Solver instance names must be filename-safe names, not paths")
        supplied_names.append(instance_name)
        run_names = _run_names(entry)
        all_run_names.extend(run_names)
        increment_seed = entry.get("increment_seed_per_replica", True)
        if not isinstance(increment_seed, bool):
            raise ValueError("increment_seed_per_replica must be true or false")
        parameters = entry.get("parameters", {})
        devices = _entry_devices(entry, parameters)
        if devices and parameters.get("device") == "auto":
            raise ValueError("Use devices instead of device: auto when defining a GPU pool")
    if len(supplied_names) != len(set(supplied_names)):
        raise ValueError("Every solver instance name must be unique")
    if len(all_run_names) != len(set(all_run_names)):
        raise ValueError("Solver instance and replica names must be unique")

    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be a YAML mapping")
    mode = execution.get("mode", "sequential")
    if mode not in {"sequential", "parallel"}:
        raise ValueError("execution.mode must be sequential or parallel")
    order = execution.get("order", "problem_first")
    if order not in {"problem_first", "solver_first"}:
        raise ValueError("execution.order must be problem_first or solver_first")
    default_cpu_workers = max(1, (os.cpu_count() or 2) // 2)
    cpu_workers = execution.get("cpu_workers", default_cpu_workers)
    if not isinstance(cpu_workers, int) or isinstance(cpu_workers, bool) or cpu_workers < 1:
        raise ValueError("execution.cpu_workers must be a positive integer")
    if mode == "parallel" and any(
        str(entry.get("parameters", {}).get("device", "")).strip().lower() == "auto"
        for entry in solver_entries
    ):
        raise ValueError(
            "device: auto is ambiguous in parallel mode; use an explicit devices list"
        )

    continue_on_error = config.get("continue_on_error", False)
    if not isinstance(continue_on_error, bool):
        raise ValueError("continue_on_error must be true or false")

    grouped: dict[Path, list[tuple[Path, Path]]] = {}
    for problem_path in problem_files:
        relative = problem_path.relative_to(problem_root)
        try:
            qubos_index = relative.parts.index("qubos")
        except ValueError as error:
            raise ValueError(f"Problem file is not inside a qubos folder: {problem_path}") from error
        if qubos_index == 0:
            raise ValueError("Every qubos folder must be inside a named batch folder")
        batch_root = problem_root.joinpath(*relative.parts[:qubos_index])
        below_qubos = Path(*relative.parts[qubos_index + 1 :])
        grouped.setdefault(batch_root, []).append((problem_path, below_qubos))

    result_roots: dict[Path, Path] = {}
    for batch_root, group in grouped.items():
        base = batch_root / "results"
        collision = any(
            (base / relative.parent / f"{relative.stem}--{name}.json").exists()
            for _, relative in group
            for name in all_run_names
        )
        result_roots[batch_root] = _available_directory(base) if collision else base

    problem_records = []
    for problem_path in problem_files:
        relative_problem = problem_path.relative_to(problem_root)
        qubos_index = relative_problem.parts.index("qubos")
        batch_root = problem_root.joinpath(*relative_problem.parts[:qubos_index])
        below_qubos = Path(*relative_problem.parts[qubos_index + 1 :])
        problem_records.append((problem_path, batch_root, below_qubos))

    tasks = []

    def add_tasks(problem_record, entry):
        problem_path, batch_root, below_qubos = problem_record
        base_parameters = entry.get("parameters", {})
        parameter_schema = describe_solver(entry["type"])["parameters"]
        devices = _entry_devices(entry, base_parameters)
        run_names = _run_names(entry)
        increment_seed = entry.get("increment_seed_per_replica", True)
        for replica_index, instance_name in enumerate(run_names):
            parameters = dict(base_parameters)
            if len(run_names) > 1 and increment_seed and "seed" in parameter_schema:
                base_seed = base_parameters.get("seed", parameter_schema["seed"].get("default", 0))
                parameters["seed"] = int(base_seed) + replica_index
            tasks.append(
                _SolveTask(
                    problem_path=problem_path,
                    solver_type=entry["type"],
                    parameters=parameters,
                    instance_name=instance_name,
                    output_path=(
                        result_roots[batch_root]
                        / below_qubos.parent
                        / f"{below_qubos.stem}--{instance_name}.json"
                    ),
                    devices=devices,
                )
            )

    if order == "problem_first":
        for problem_record in problem_records:
            for entry in solver_entries:
                add_tasks(problem_record, entry)
    else:
        for entry in solver_entries:
            for problem_record in problem_records:
                add_tasks(problem_record, entry)

    print(
        f"Execution: {mode}, {order}, {len(tasks)} task(s), "
        f"CPU workers={cpu_workers}"
    )
    if mode == "parallel":
        saved, failures = _run_tasks_parallel(tasks, continue_on_error, cpu_workers)
    else:
        saved, failures = _run_tasks_sequential(tasks, continue_on_error)
    destinations = ", ".join(str(value) for value in result_roots.values())
    print(f"\nSaved {len(saved)} result(s) in {destinations}; {failures} failed or skipped.")
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
