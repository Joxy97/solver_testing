"""Run a resumable four-way relative QUBO benchmark.

The benchmark deliberately excludes exact enumeration and treats simulated
bifurcation on CPU and CUDA as separate solver configurations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from problems.manager import load_problem
from solvers.manager import solve_problem


SOLVERS = {
    "scipy_highs": {
        "solver_type": "scipy_highs",
        "label": "SciPy + HiGHS",
        # These are capacity gates, not optimization tuning. They permit every
        # 1,000-variable QUBO, including a fully dense one.
        "parameters": {
            "max_quadratic_terms": 500_000,
            "max_linearized_variables": 501_000,
            # Dense linearizations can remain inside HiGHS presolve far beyond
            # its timer. Disabling it gives more predictable benchmark timing.
            "presolve": False,
        },
    },
    "ocean_sa": {
        "solver_type": "ocean_sa",
        "label": "Ocean Simulated Annealing (CPU)",
        "parameters": {},
    },
    "bifurcation_cpu": {
        "solver_type": "simulated_bifurcation",
        "label": "Simulated Bifurcation (CPU)",
        "parameters": {"device": "cpu"},
    },
    "bifurcation_cuda": {
        "solver_type": "simulated_bifurcation",
        "label": "Simulated Bifurcation (CUDA)",
        "parameters": {"device": "cuda"},
    },
}

PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "highspy",
    "dwave-ocean-sdk",
    "dwave-samplers",
    "torch",
    "simulated-bifurcation",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def expand_problem_paths(inputs: list[str]) -> list[Path]:
    import glob

    paths: set[Path] = set()
    for item in inputs:
        candidate = Path(item)
        if candidate.is_dir():
            paths.update(path.resolve() for path in candidate.rglob("*.json"))
        elif any(character in item for character in "*?["):
            paths.update(Path(match).resolve() for match in glob.glob(item, recursive=True))
        elif candidate.is_file():
            paths.add(candidate.resolve())
        else:
            raise FileNotFoundError(f"Problem input does not exist or match files: {item}")
    if not paths:
        raise ValueError("No problem JSON files were found")
    return sorted(paths)


def load_solver_config(path: Path | None, snapshots: int | None) -> dict:
    config = json.loads(json.dumps(SOLVERS))
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle) or {}
        if not isinstance(supplied, dict) or not isinstance(supplied.get("solvers", {}), dict):
            raise ValueError("Benchmark config must contain a 'solvers' mapping")
        unknown = set(supplied.get("solvers", {})).difference(config)
        if unknown:
            raise ValueError(f"Unknown benchmark solver aliases: {', '.join(sorted(unknown))}")
        for alias, parameters in supplied.get("solvers", {}).items():
            if not isinstance(parameters, dict):
                raise ValueError(f"Configuration for {alias} must be a mapping")
            config[alias]["parameters"].update(parameters)
    if snapshots is not None:
        if snapshots < 0:
            raise ValueError("--snapshots cannot be negative")
        for entry in config.values():
            entry["parameters"]["num_snapshots"] = snapshots
    return config


def benchmark_identity(problems: list[dict], solvers: dict) -> str:
    identity = {
        "instances": [problem["instance_id"] for problem in problems],
        "solvers": solvers,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def preflight(require_cuda: bool) -> dict:
    errors = []
    details = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {name: package_version(name) for name in PACKAGE_NAMES},
        "cuda_available": None,
        "cuda_device_count": None,
        "cuda_devices": [],
    }
    required = ["scipy", "highspy", "dwave-samplers", "torch", "simulated-bifurcation"]
    missing = [name for name in required if details["packages"][name] is None]
    if missing:
        errors.append("Missing Python distributions: " + ", ".join(missing))
    try:
        import torch

        details["cuda_available"] = bool(torch.cuda.is_available())
        details["cuda_device_count"] = int(torch.cuda.device_count())
        details["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if require_cuda and not torch.cuda.is_available():
            errors.append("CUDA is unavailable to PyTorch, but the four-way benchmark requires it")
    except ImportError:
        if "torch" not in missing:
            errors.append("PyTorch could not be imported")
    details["errors"] = errors
    return details


def create_progress(benchmark_id: str, problems: list[dict], paths: list[Path], solvers: dict) -> dict:
    tasks = {}
    for problem, path in zip(problems, paths):
        for alias in solvers:
            key = f"{problem['instance_id']}::{alias}"
            tasks[key] = {
                "instance_id": problem["instance_id"],
                "problem_path": str(path),
                "solver_alias": alias,
                "status": "pending",
                "attempts": 0,
                "elapsed_seconds": None,
                "result_path": None,
                "error": None,
            }
    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "tasks": tasks,
    }


def load_or_create_progress(
    path: Path,
    benchmark_id: str,
    problems: list[dict],
    problem_paths: list[Path],
    solvers: dict,
    resume: bool,
) -> dict:
    if path.exists() and resume:
        with path.open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        if progress.get("benchmark_id") != benchmark_id:
            raise RuntimeError(
                "Existing progress.json belongs to different inputs or solver settings. "
                "Choose another --output directory."
            )
        for task in progress["tasks"].values():
            if task["status"] == "running":
                task["status"] = "pending"
        return progress
    if path.exists() and not resume:
        raise RuntimeError(
            "Output already contains progress.json. Use the default resume behavior or "
            "choose a new --output directory."
        )
    return create_progress(benchmark_id, problems, problem_paths, solvers)


def update_progress(path: Path, progress: dict) -> None:
    progress["updated_at"] = utc_now()
    atomic_json(path, progress)


def progress_line(progress: dict, current: str = "") -> str:
    tasks = list(progress["tasks"].values())
    completed = sum(task["status"] in ("completed", "failed") for task in tasks)
    successful = sum(task["status"] == "completed" for task in tasks)
    failed = sum(task["status"] == "failed" for task in tasks)
    durations = [
        float(task["elapsed_seconds"])
        for task in tasks
        if task["status"] in ("completed", "failed") and task["elapsed_seconds"] is not None
    ]
    eta = statistics.mean(durations) * (len(tasks) - completed) if durations else None
    width = 28
    filled = round(width * completed / len(tasks))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {completed}/{len(tasks)} ({100 * completed / len(tasks):5.1f}%) "
        f"ok={successful} failed={failed} ETA={format_seconds(eta)} {current}"
    ).rstrip()


def close_energy(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-9)


def hamming(first: list[int], second: list[int]) -> int:
    return sum(left != right for left, right in zip(first, second))


def build_instance_report(instance_id: str, result_paths: dict[str, Path], failures: dict, solvers: dict) -> dict:
    results = {}
    for alias, path in result_paths.items():
        with path.open("r", encoding="utf-8") as handle:
            results[alias] = json.load(handle)

    energies = {alias: float(result["solution"]["energy"]) for alias, result in results.items()}
    best_energy = min(energies.values()) if energies else None
    best_aliases = (
        [alias for alias, energy in energies.items() if close_energy(energy, best_energy)]
        if best_energy is not None
        else []
    )
    rows = []
    for alias in solvers:
        if alias not in results:
            rows.append(
                {
                    "solver_alias": alias,
                    "solver": solvers[alias]["label"],
                    "completed": False,
                    "error": failures.get(alias),
                }
            )
            continue
        result = results[alias]
        energy = energies[alias]
        better = sum(
            other < energy and not close_energy(other, energy) for other in energies.values()
        )
        winner_samples = [results[winner]["solution"]["sample"] for winner in best_aliases]
        sample = result["solution"]["sample"]
        distance_to_best = min(hamming(sample, winner) for winner in winner_samples)
        gap = energy - best_energy
        rows.append(
            {
                "solver_alias": alias,
                "solver": solvers[alias]["label"],
                "completed": True,
                "status": result["status"],
                "energy": energy,
                "rank": 1 + better,
                "is_best_or_tied": alias in best_aliases,
                "absolute_gap_to_best": gap,
                "relative_gap_to_best": gap / max(1.0, abs(best_energy)),
                "hamming_to_nearest_best": distance_to_best,
                "hamming_fraction_to_nearest_best": distance_to_best / len(sample),
                "sample_ones": sum(sample),
                "wall_seconds": float(result["timing"]["wall_seconds"]),
                "device": result["solver"]["device"],
                "verification_passed": bool(result["verification"]["passed"]),
                "result_path": str(result_paths[alias]),
            }
        )

    pairwise_hamming = {}
    pairwise_energy = {}
    for left in solvers:
        pairwise_hamming[left] = {}
        pairwise_energy[left] = {}
        for right in solvers:
            if left in results and right in results:
                distance = hamming(
                    results[left]["solution"]["sample"],
                    results[right]["solution"]["sample"],
                )
                pairwise_hamming[left][right] = {
                    "distance": distance,
                    "fraction": distance / len(results[left]["solution"]["sample"]),
                }
                pairwise_energy[left][right] = energies[left] - energies[right]
            else:
                pairwise_hamming[left][right] = None
                pairwise_energy[left][right] = None
    any_result = next(iter(results.values()), None)
    return {
        "schema_version": 1,
        "instance_id": instance_id,
        "problem": any_result["problem"] if any_result else None,
        "complete_four_way_comparison": len(results) == len(solvers),
        "best_energy": best_energy,
        "best_solver_aliases": best_aliases,
        "solver_results": rows,
        "pairwise_hamming": pairwise_hamming,
        "pairwise_energy_difference_left_minus_right": pairwise_energy,
    }


def instance_markdown(report: dict, solvers: dict) -> str:
    lines = [
        f"# Instance report: {report['instance_id']}",
        "",
        f"- Complete four-way comparison: {'yes' if report['complete_four_way_comparison'] else 'no'}",
        f"- Best energy: {report['best_energy'] if report['best_energy'] is not None else 'unavailable'}",
        f"- Best solver(s): {', '.join(report['best_solver_aliases']) or 'none'}",
        "",
        "## Solver comparison",
        "",
        "| Solver | Completed | Energy | Rank | Gap to best | Relative gap | Wall s | Device |",
        "|---|:---:|---:|---:|---:|---:|---:|---|",
    ]
    row_by_alias = {row["solver_alias"]: row for row in report["solver_results"]}
    for alias in solvers:
        row = row_by_alias[alias]
        if not row["completed"]:
            lines.append(f"| {row['solver']} | no | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {row['solver']} | yes | {row['energy']:.12g} | {row['rank']} | "
                f"{row['absolute_gap_to_best']:.12g} | {row['relative_gap_to_best']:.6%} | "
                f"{row['wall_seconds']:.3f} | {row['device']} |"
            )
    lines.extend(
        [
            "",
            "## Pairwise Hamming distance",
            "",
            "| Solver | " + " | ".join(solvers) + " |",
            "|---|" + "---:|" * len(solvers),
        ]
    )
    for left in solvers:
        values = []
        for right in solvers:
            item = report["pairwise_hamming"][left][right]
            values.append(str(item["distance"]) if item is not None else "—")
        lines.append(f"| {left} | " + " | ".join(values) + " |")
    failures = [row for row in report["solver_results"] if not row["completed"]]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(f"- `{row['solver_alias']}`: {row.get('error') or 'unknown error'}")
    return "\n".join(lines) + "\n"


def write_instance_report(output: Path, report: dict, solvers: dict) -> None:
    base = output / "instances" / report["instance_id"]
    atomic_json(base.with_suffix(".json"), report)
    atomic_text(base.with_suffix(".md"), instance_markdown(report, solvers))
    fields = [
        "solver_alias", "solver", "completed", "status", "energy", "rank",
        "is_best_or_tied", "absolute_gap_to_best", "relative_gap_to_best",
        "hamming_to_nearest_best", "hamming_fraction_to_nearest_best", "sample_ones",
        "wall_seconds", "device", "verification_passed", "result_path", "error",
    ]
    write_csv(base.with_suffix(".csv"), report["solver_results"], fields)


def aggregate_reports(reports: list[dict], solvers: dict) -> tuple[dict, list[dict]]:
    complete = [report for report in reports if report["complete_four_way_comparison"]]
    rows = []
    for alias in solvers:
        all_rows = [
            row
            for report in reports
            for row in report["solver_results"]
            if row["solver_alias"] == alias and row["completed"]
        ]
        fair_rows = [
            row
            for report in complete
            for row in report["solver_results"]
            if row["solver_alias"] == alias
        ]
        energies = [row["energy"] for row in fair_rows]
        walls = [row["wall_seconds"] for row in fair_rows]
        rows.append(
            {
                "solver_alias": alias,
                "solver": solvers[alias]["label"],
                "completed_runs": len(all_rows),
                "failed_runs": len(reports) - len(all_rows),
                "complete_case_instances": len(fair_rows),
                "wins_or_ties": sum(row["is_best_or_tied"] for row in fair_rows),
                "win_or_tie_rate": (
                    statistics.mean(row["is_best_or_tied"] for row in fair_rows)
                    if fair_rows else None
                ),
                "mean_rank": statistics.mean(row["rank"] for row in fair_rows) if fair_rows else None,
                "mean_energy": statistics.mean(energies) if energies else None,
                "sd_energy": statistics.stdev(energies) if len(energies) > 1 else None,
                "mean_absolute_gap_to_best": (
                    statistics.mean(row["absolute_gap_to_best"] for row in fair_rows)
                    if fair_rows else None
                ),
                "mean_relative_gap_to_best": (
                    statistics.mean(row["relative_gap_to_best"] for row in fair_rows)
                    if fair_rows else None
                ),
                "mean_hamming_fraction_to_nearest_best": (
                    statistics.mean(row["hamming_fraction_to_nearest_best"] for row in fair_rows)
                    if fair_rows else None
                ),
                "mean_wall_seconds": statistics.mean(walls) if walls else None,
                "median_wall_seconds": statistics.median(walls) if walls else None,
                "sd_wall_seconds": statistics.stdev(walls) if len(walls) > 1 else None,
                "total_wall_seconds": sum(row["wall_seconds"] for row in all_rows),
            }
        )

    pairwise = {}
    aliases = list(solvers)
    for left in aliases:
        pairwise[left] = {}
        for right in aliases:
            wins = ties = losses = 0
            for report in complete:
                energy_by_alias = {
                    row["solver_alias"]: row["energy"] for row in report["solver_results"]
                }
                left_energy = energy_by_alias[left]
                right_energy = energy_by_alias[right]
                if close_energy(left_energy, right_energy):
                    ties += 1
                elif left_energy < right_energy:
                    wins += 1
                else:
                    losses += 1
            pairwise[left][right] = {"wins": wins, "ties": ties, "losses": losses}
    aggregate = {
        "schema_version": 1,
        "instances_requested": len(reports),
        "complete_four_way_instances": len(complete),
        "solver_summary": rows,
        "pairwise_energy_outcomes": pairwise,
    }
    return aggregate, rows


def aggregate_markdown(aggregate: dict, solvers: dict) -> str:
    lines = [
        "# Four-solver relative QUBO benchmark",
        "",
        f"- Instances requested: {aggregate['instances_requested']}",
        f"- Complete four-way comparisons: {aggregate['complete_four_way_instances']}",
        "- Exact enumeration: excluded",
        "- Comparative averages use complete four-way instances only.",
        "",
        "## Average results",
        "",
        "| Solver | Runs | Wins/ties | Mean rank | Mean energy | Mean gap | Mean relative gap | Mean wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["solver_summary"]:
        def number(key: str, digits: int = 4):
            value = row[key]
            return "—" if value is None else f"{value:.{digits}f}"

        relative = (
            "—" if row["mean_relative_gap_to_best"] is None
            else f"{row['mean_relative_gap_to_best']:.4%}"
        )
        lines.append(
            f"| {row['solver']} | {row['completed_runs']} | {row['wins_or_ties']} | "
            f"{number('mean_rank', 3)} | {number('mean_energy', 6)} | "
            f"{number('mean_absolute_gap_to_best', 6)} | {relative} | "
            f"{number('mean_wall_seconds', 3)} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise energy outcomes",
            "",
            "Each cell is `wins/ties/losses` for the row solver against the column solver.",
            "",
            "| Solver | " + " | ".join(solvers) + " |",
            "|---|" + "---:|" * len(solvers),
        ]
    )
    for left in solvers:
        values = []
        for right in solvers:
            item = aggregate["pairwise_energy_outcomes"][left][right]
            values.append(f"{item['wins']}/{item['ties']}/{item['losses']}")
        lines.append(f"| {left} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower energy is better.",
            "- Rank and gap metrics are calculated within each instance before averaging.",
            "- A win includes an energy tie within numerical tolerance.",
            "- Runtime is wall-clock time measured around the verified solver adapter.",
            "- CPU and CUDA bifurcation use the same algorithmic defaults but are reported separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def collect_reports(output: Path, problems: list[dict], solvers: dict, progress: dict) -> list[dict]:
    reports = []
    for problem in problems:
        instance_id = problem["instance_id"]
        result_paths = {}
        failures = {}
        for alias in solvers:
            task = progress["tasks"][f"{instance_id}::{alias}"]
            if task["status"] == "completed" and task["result_path"]:
                result_paths[alias] = Path(task["result_path"])
            elif task["status"] == "failed":
                failures[alias] = task.get("error")
        report = build_instance_report(instance_id, result_paths, failures, solvers)
        write_instance_report(output, report, solvers)
        reports.append(report)
    return reports


def export_aggregate(output: Path, reports: list[dict], solvers: dict) -> None:
    aggregate, rows = aggregate_reports(reports, solvers)
    atomic_json(output / "aggregate_report.json", aggregate)
    atomic_text(output / "aggregate_report.md", aggregate_markdown(aggregate, solvers))
    write_csv(output / "aggregate_report.csv", rows, list(rows[0]))

    flat = []
    for report in reports:
        for row in report["solver_results"]:
            flat.append({"instance_id": report["instance_id"], **row})
    fields = ["instance_id"] + sorted({key for row in flat for key in row if key != "instance_id"})
    write_csv(output / "all_solver_results.csv", flat, fields)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Benchmark SciPy+HiGHS, Ocean SA, and simulated bifurcation on CPU and CUDA. "
            "Exact enumeration is never used."
        )
    )
    result.add_argument(
        "problems",
        nargs="*",
        default=["generated_problems"],
        help="Problem JSON files, glob patterns, or directories (default: generated_problems)",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("BenchmarkResults/relative_4_solver"),
        help="Benchmark output directory",
    )
    result.add_argument(
        "--config",
        type=Path,
        help="Optional YAML parameter overrides keyed by the four solver aliases",
    )
    result.add_argument(
        "--snapshots",
        type=int,
        help="Set num_snapshots for all solvers (default: each solver's zero-snapshot default)",
    )
    result.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse an existing progress file instead of resuming it",
    )
    result.add_argument(
        "--allow-missing-solvers",
        action="store_true",
        help="Record solver failures and produce partial reports instead of failing preflight",
    )
    result.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first solver failure (progress remains resumable)",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the task plan without importing or running solvers",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    problem_paths = expand_problem_paths(args.problems)
    problems = [load_problem(path) for path in problem_paths]
    instance_ids = [problem["instance_id"] for problem in problems]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Problem inputs contain duplicate instance IDs")
    solvers = load_solver_config(args.config, args.snapshots)
    identity = benchmark_identity(problems, solvers)

    print(f"Instances: {len(problems)}")
    print(f"Solver configurations: {len(solvers)} ({', '.join(solvers)})")
    print(f"Planned solver runs: {len(problems) * len(solvers)}")
    print(f"Output: {args.output.resolve()}")
    if args.dry_run:
        print("Dry run complete; no solver was imported or executed.")
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    environment = preflight(require_cuda=not args.allow_missing_solvers)
    atomic_json(args.output / "environment.json", environment)
    if environment["errors"] and not args.allow_missing_solvers:
        raise RuntimeError("; ".join(environment["errors"]))

    resolved = {
        "schema_version": 1,
        "benchmark_id": identity,
        "created_at": utc_now(),
        "problem_paths": [str(path) for path in problem_paths],
        "solvers": solvers,
        "notes": [
            "Exact enumeration is excluded.",
            "Bifurcation CPU and CUDA are independent benchmark configurations.",
            "Comparative aggregate metrics use only instances completed by all four solvers.",
        ],
    }
    progress_path = args.output / "progress.json"
    progress = load_or_create_progress(
        progress_path,
        identity,
        problems,
        problem_paths,
        solvers,
        resume=not args.no_resume,
    )
    atomic_json(args.output / "benchmark_config.json", resolved)
    update_progress(progress_path, progress)
    print(progress_line(progress))

    interrupted = False
    try:
        for problem, problem_path in zip(problems, problem_paths):
            instance_id = problem["instance_id"]
            for alias, entry in solvers.items():
                key = f"{instance_id}::{alias}"
                task = progress["tasks"][key]
                if task["status"] == "completed" and task.get("result_path"):
                    if Path(task["result_path"]).is_file():
                        print(progress_line(progress, f"resume-skip {instance_id} / {alias}"))
                        continue
                    task["status"] = "pending"
                task.update(
                    {
                        "status": "running",
                        "attempts": int(task.get("attempts", 0)) + 1,
                        "started_at": utc_now(),
                        "finished_at": None,
                        "error": None,
                    }
                )
                update_progress(progress_path, progress)
                print(progress_line(progress, f"running {instance_id} / {alias}"), flush=True)
                started = time.perf_counter()
                try:
                    result = solve_problem(
                        problem,
                        entry["solver_type"],
                        entry["parameters"],
                        source=problem_path,
                    )
                    result["benchmark_solver_alias"] = alias
                    result_path = args.output / "raw" / instance_id / f"{alias}.json"
                    atomic_json(result_path, result)
                    task.update(
                        {
                            "status": "completed",
                            "elapsed_seconds": time.perf_counter() - started,
                            "finished_at": utc_now(),
                            "result_path": str(result_path.resolve()),
                        }
                    )
                except Exception as error:  # Keep the remaining benchmark resumable.
                    failure = {
                        "instance_id": instance_id,
                        "solver_alias": alias,
                        "solver_type": entry["solver_type"],
                        "parameters": entry["parameters"],
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "occurred_at": utc_now(),
                    }
                    failure_path = args.output / "failures" / f"{instance_id}--{alias}.json"
                    atomic_json(failure_path, failure)
                    task.update(
                        {
                            "status": "failed",
                            "elapsed_seconds": time.perf_counter() - started,
                            "finished_at": utc_now(),
                            "error": f"{type(error).__name__}: {error}",
                            "failure_path": str(failure_path.resolve()),
                        }
                    )
                finally:
                    update_progress(progress_path, progress)
                    print(progress_line(progress, f"finished {instance_id} / {alias}"), flush=True)

                # Refresh the per-instance report as soon as all four attempts finish.
                instance_tasks = [
                    progress["tasks"][f"{instance_id}::{solver_alias}"] for solver_alias in solvers
                ]
                if all(item["status"] in ("completed", "failed") for item in instance_tasks):
                    collect_reports(args.output, [problem], solvers, progress)
                if task["status"] == "failed" and args.fail_fast:
                    raise RuntimeError(task["error"])
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Progress was saved and can be resumed with the same command.")
    finally:
        for task in progress["tasks"].values():
            if task["status"] == "running":
                task["status"] = "pending"
        update_progress(progress_path, progress)
        reports = collect_reports(args.output, problems, solvers, progress)
        export_aggregate(args.output, reports, solvers)

    failed = sum(task["status"] == "failed" for task in progress["tasks"].values())
    pending = sum(task["status"] == "pending" for task in progress["tasks"].values())
    print(progress_line(progress))
    print(f"Aggregate report: {(args.output / 'aggregate_report.md').resolve()}")
    if interrupted or pending:
        return 130 if interrupted else 2
    if failed and not args.allow_missing_solvers:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
