"""Combine leaf manifests in a generated dataset into a root catalog and summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


BASE_FIELDS = [
    "study",
    "cell",
    "json_path",
    "manifest_path",
    "instance_id",
    "problem_type",
    "problem_name",
    "seed",
    "num_variables",
    "linear_terms",
    "quadratic_terms",
    "validation_status",
    "warning_codes",
    "error_codes",
]


def build_catalog(dataset_root: Path) -> tuple[Path, Path]:
    dataset_root = dataset_root.resolve()
    manifests = sorted(
        path for path in dataset_root.rglob("manifest.csv") if path.parent != dataset_root
    )
    if not manifests:
        raise ValueError(f"No leaf manifest.csv files found below {dataset_root}")

    rows = []
    seen_ids = set()
    missing_files = []
    for manifest in manifests:
        cell = manifest.parent.relative_to(dataset_root).as_posix()
        study = cell.split("/", 1)[0]
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                instance_id = source["instance_id"]
                if instance_id in seen_ids:
                    raise ValueError(f"Duplicate instance ID in dataset: {instance_id}")
                seen_ids.add(instance_id)
                json_file = manifest.parent / source["file"]
                if not json_file.is_file():
                    missing_files.append(str(json_file))
                row = dict(source)
                row.update(
                    {
                        "study": study,
                        "cell": cell,
                        "json_path": json_file.relative_to(dataset_root).as_posix(),
                        "manifest_path": manifest.relative_to(dataset_root).as_posix(),
                    }
                )
                row.pop("file", None)
                rows.append(row)
    if missing_files:
        raise ValueError("Catalog references missing JSON files: " + ", ".join(missing_files))

    extra_fields = sorted(
        {key for row in rows for key in row if key not in BASE_FIELDS}
    )
    catalog_path = dataset_root / "catalog.csv"
    temporary_catalog = catalog_path.with_suffix(".csv.tmp")
    with temporary_catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_FIELDS + extra_fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["cell"], row["instance_id"])))
    temporary_catalog.replace(catalog_path)

    status_counts = Counter(row["validation_status"] for row in rows)
    warning_counts = Counter(
        warning
        for row in rows
        for warning in row.get("warning_codes", "").split(";")
        if warning
    )
    study_counts = Counter(row["study"] for row in rows)
    study_cells = Counter(
        study for study, _ in {(row["study"], row["cell"]) for row in rows}
    )
    json_bytes = sum((dataset_root / row["json_path"]).stat().st_size for row in rows)
    summary = {
        "dataset": dataset_root.name,
        "problem_types": sorted({row["problem_type"] for row in rows}),
        "total_cells": len(manifests),
        "total_instances": len(rows),
        "json_storage_bytes": json_bytes,
        "json_storage_mib": round(json_bytes / 1024**2, 3),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "warning_code_counts": dict(sorted(warning_counts.items())),
        "study_counts": {
            study: {
                "cells": study_cells[study],
                "instances": study_counts[study],
            }
            for study in sorted(study_counts)
        },
        "num_variables": sorted({int(row["num_variables"]) for row in rows}),
        "quadratic_densities": sorted(
            {float(row["parameter.quadratic_density"]) for row in rows}
        ),
        "distributions": sorted({row["parameter.distribution"] for row in rows}),
        "catalog": catalog_path.name,
        "specification": "dataset_spec.yaml",
    }
    summary_path = dataset_root / "summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    with temporary_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    temporary_summary.replace(summary_path)
    return catalog_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path("generated_problems/random_qubo"),
    )
    args = parser.parse_args()
    catalog, summary = build_catalog(args.dataset_root)
    print(f"Catalog: {catalog}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
