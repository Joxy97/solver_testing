"""Merge seed-level 100k QUBO runs into dashboard-ready density experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import tarfile
from datetime import datetime, timezone
from pathlib import Path


DENSITIES = (
    ("0p01", 0.01),
    ("0p10", 0.10),
    ("0p25", 0.25),
    ("0p50", 0.50),
    ("0p75", 0.75),
    ("1p00", 1.00),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def arithmetic_mean(rows: list[dict], field: str):
    values = [row[field] for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def merge_density(source: Path, destination: Path, tag: str, density: float) -> dict:
    run_directories = sorted(
        (path for path in source.glob("seed_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("seed_")),
    )
    if [path.name for path in run_directories] != [f"seed_{seed}" for seed in range(10)]:
        raise RuntimeError(f"{tag}: expected complete seed_0 through seed_9 run directories")

    configurations = [read_json(path / "benchmark_config.json") for path in run_directories]
    solver_definitions = configurations[0]["solvers"]
    devices = configurations[0]["devices"]
    for configuration in configurations[1:]:
        if configuration["solvers"] != solver_definitions:
            raise RuntimeError(f"{tag}: solver definitions differ between seeds")
        if configuration["devices"] != devices:
            raise RuntimeError(f"{tag}: CUDA device selections differ between seeds")

    identity = {
        "density": density,
        "seeds": list(range(10)),
        "solvers": solver_definitions,
        "devices": devices,
    }
    benchmark_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("failures", "instances", "raw"):
        (destination / name).mkdir(parents=True, exist_ok=True)

    instance_reports = []
    all_rows = []
    progress_tasks = {}
    failure_count = 0
    created_at = min(configuration["created_at"] for configuration in configurations)
    updated_at = created_at

    for seed, run in enumerate(run_directories):
        reports = list((run / "instances").glob("*.json"))
        if len(reports) != 1:
            raise RuntimeError(f"{tag}/{run.name}: expected exactly one instance report")
        report = read_json(reports[0])
        instance_id = report["instance_id"]
        instance_reports.append(report)
        all_rows.extend(
            {"density": density, "seed": seed, **row}
            for row in report["solver_results"]
        )
        shutil.copy2(reports[0], destination / "instances" / reports[0].name)
        for suffix in (".csv", ".md"):
            sibling = reports[0].with_suffix(suffix)
            if sibling.is_file():
                shutil.copy2(sibling, destination / "instances" / sibling.name)

        raw_source = run / "raw" / instance_id
        raw_destination = destination / "raw" / instance_id
        shutil.copytree(raw_source, raw_destination, dirs_exist_ok=True)

        progress = read_json(run / "progress.json")
        updated_at = max(updated_at, progress["updated_at"])
        for key, task in progress["tasks"].items():
            if key in progress_tasks:
                raise RuntimeError(f"{tag}: duplicate task key {key}")
            progress_tasks[key] = task

        for failure in (run / "failures").glob("*.json"):
            failure_count += 1
            shutil.copy2(failure, destination / "failures" / failure.name)

    aliases = list(solver_definitions)
    summaries = []
    for alias in aliases:
        rows = [row for row in all_rows if row["solver_alias"] == alias]
        completed = [row for row in rows if row.get("completed")]
        summaries.append(
            {
                "solver_alias": alias,
                "solver": solver_definitions[alias]["label"],
                "completed_runs": len(completed),
                "failed_runs": len(rows) - len(completed),
                "wins_or_ties": sum(bool(row.get("is_best_or_tied")) for row in completed),
                "mean_rank": arithmetic_mean(completed, "rank"),
                "mean_energy": arithmetic_mean(completed, "energy"),
                "mean_absolute_gap_to_best": arithmetic_mean(
                    completed, "absolute_gap_to_best"
                ),
                "mean_relative_gap_to_best": arithmetic_mean(
                    completed, "relative_gap_to_best"
                ),
                "mean_wall_seconds": arithmetic_mean(completed, "wall_seconds"),
            }
        )

    complete_comparisons = sum(
        bool(report.get("complete_comparison")) for report in instance_reports
    )
    aggregate = {
        "schema_version": 2,
        "instances_requested": 10,
        "complete_comparisons": complete_comparisons,
        "solver_summary": summaries,
    }
    base_problem = {**configurations[0]["problem"]}
    base_problem.pop("seed", None)
    config = {
        "schema_version": 2,
        "benchmark_type": "implicit_random_qubo_density_8gpu_10_seed",
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "problem": {**base_problem, "seeds": list(range(10))},
        "solvers": solver_definitions,
        "devices": devices,
        "notes": [
            f"Ten deterministic 100,000-variable Random QUBOs at density {density:.2f}.",
            "Each solver configuration was fixed after a separate seed-0 tuning round.",
            "Every aggregate result is the best of eight CUDA-device trajectory shards.",
            "Final objectives use deterministic coefficient regeneration and int64 accumulation.",
        ],
    }
    environment = read_json(run_directories[0] / "environment.json")
    progress = {
        "schema_version": 2,
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "tasks": progress_tasks,
    }
    write_json(destination / "benchmark_config.json", config)
    write_json(destination / "environment.json", environment)
    write_json(destination / "progress.json", progress)
    write_json(destination / "aggregate_report.json", aggregate)
    write_csv(destination / "aggregate_report.csv", summaries, list(summaries[0]))
    write_csv(destination / "all_solver_results.csv", all_rows, list(all_rows[0]))

    lines = [
        f"# 100k Random QUBO density {density:.2f}",
        "",
        f"- Complete comparisons: {complete_comparisons} / 10",
        f"- Device task failures: {failure_count}",
        "",
        "| Solver | Runs | Wins/ties | Mean energy | Mean gap | Mean wall s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['solver']} | {summary['completed_runs']} | "
            f"{summary['wins_or_ties']} | {summary['mean_energy']:.3f} | "
            f"{summary['mean_relative_gap_to_best']:.6f} | "
            f"{summary['mean_wall_seconds']:.3f} |"
        )
    write_text(destination / "aggregate_report.md", "\n".join(lines) + "\n")
    return {
        "tag": tag,
        "quadratic_density": density,
        "path": tag,
        "benchmark_id": benchmark_id,
        "instances": len(instance_reports),
        "complete_comparisons": complete_comparisons,
        "failure_records": failure_count,
        "solver_summary": summaries,
    }


def archive_device_runs(source: Path, destination: Path) -> None:
    archive = destination / "device_runs.tar.gz"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as handle:
        run_directories = sorted(
            (path for path in source.glob("seed_*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("seed_")),
        )
        for run in run_directories:
            handle.add(run / "device_runs", arcname=f"{run.name}/device_runs")
    temporary.replace(archive)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Root containing TAG/seed_N run folders")
    parser.add_argument("output", type=Path, help="Dashboard-ready sweep output root")
    parser.add_argument("--archive-device-runs", action="store_true")
    args = parser.parse_args()

    summaries = []
    for tag, density in DENSITIES:
        source = args.source / tag
        destination = args.output / tag
        summary = merge_density(source, destination, tag, density)
        if args.archive_device_runs:
            archive_device_runs(source, destination)
        summaries.append(summary)
        print(
            f"{tag}: {summary['complete_comparisons']}/10 complete, "
            f"{summary['failure_records']} failures"
        )

    manifest = {
        "schema_version": 1,
        "benchmark_type": "implicit_random_qubo_density_sweep_8gpu",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "densities": summaries,
    }
    write_json(args.output / "suite_manifest.json", manifest)
    write_csv(
        args.output / "suite_index.csv",
        [
            {
                "tag": item["tag"],
                "quadratic_density": item["quadratic_density"],
                "instances": item["instances"],
                "complete_comparisons": item["complete_comparisons"],
                "failure_records": item["failure_records"],
                "path": item["path"],
            }
            for item in summaries
        ],
        [
            "tag",
            "quadratic_density",
            "instances",
            "complete_comparisons",
            "failure_records",
            "path",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
