"""Generate an implicit 100k-variable Random QUBO and benchmark SB against TRF.

The ordinary JSON QUBO format is intentionally not used: at density 0.25 the problem
contains about 1.25 billion quadratic terms.  A compact seed-defined manifest is saved,
and every worker materializes the identical matrix directly on its assigned CUDA device.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = Path("BenchmarkResults/random_qubo_100k_0p25_8gpu")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def problem_id(specification: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"implicit_random_qubo-{digest}"


def parse_devices(text: str | None, expected: int, available: int) -> list[int]:
    devices = list(range(expected)) if text is None else [int(item) for item in text.split(",")]
    if len(devices) != expected or len(set(devices)) != len(devices):
        raise ValueError(f"Expected {expected} distinct CUDA device indices")
    invalid = [device for device in devices if device < 0 or device >= available]
    if invalid:
        raise ValueError(
            f"CUDA device indices are unavailable: {invalid}; visible count is {available}"
        )
    return devices


def agents_for_device(total: int, count: int, position: int, mode: str) -> int:
    if mode == "replicate":
        return total
    quotient, remainder = divmod(total, count)
    result = quotient + (position < remainder)
    if result < 1:
        raise ValueError(f"Cannot shard {total} agents across {count} CUDA devices")
    return result


def _problem_blocks(torch, device: str, specification: dict, block_rows: int):
    """Yield deterministic integer linear, diagonal, and rectangular coefficient blocks."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(specification["seed"]))
    n = int(specification["num_variables"])
    low = int(specification["coefficient_min"])
    high = int(specification["coefficient_max"]) + 1
    density = float(specification["quadratic_density"])

    linear = torch.randint(low, high, (n,), dtype=torch.int8, device=device, generator=generator)
    yield "linear", 0, n, linear, n, int(torch.count_nonzero(linear).item())

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        width = stop - start

        diagonal = torch.randint(
            low, high, (width, width), dtype=torch.int8, device=device, generator=generator
        )
        diagonal_mask = torch.rand(
            (width, width), dtype=torch.float32, device=device, generator=generator
        ) < density
        diagonal_mask = torch.triu(diagonal_mask, diagonal=1)
        diagonal.masked_fill_(~diagonal_mask, 0)
        diagonal = torch.triu(diagonal, diagonal=1)
        yield (
            "diagonal",
            start,
            stop,
            diagonal,
            int(torch.count_nonzero(diagonal_mask).item()),
            int(torch.count_nonzero(diagonal).item()),
        )

        if stop < n:
            rectangle = torch.randint(
                low,
                high,
                (width, n - stop),
                dtype=torch.int8,
                device=device,
                generator=generator,
            )
            rectangle_mask = torch.rand(
                (width, n - stop),
                dtype=torch.float32,
                device=device,
                generator=generator,
            ) < density
            rectangle.masked_fill_(~rectangle_mask, 0)
            yield (
                "rectangle",
                start,
                stop,
                rectangle,
                int(torch.count_nonzero(rectangle_mask).item()),
                int(torch.count_nonzero(rectangle).item()),
            )


def materialize_problem(torch, device, specification, block_rows, representation):
    """Materialize either the augmented SB tensor or normalized TRF coupling matrix."""
    n = int(specification["num_variables"])
    matrix_size = n + 1 if representation == "bifurcation" else n
    matrix = torch.empty((matrix_size, matrix_size), dtype=torch.float32, device=device)
    h = torch.empty(n, dtype=torch.float32, device=device)
    row_absolute_sum = (
        torch.zeros(n, dtype=torch.float32, device=device)
        if representation == "transverse_route"
        else None
    )
    constant_quarters = torch.zeros((), dtype=torch.int64, device=device)
    requested_terms = 0
    stored_terms = 0
    linear_nonzero = 0

    for kind, start, stop, coefficients, requested, stored in _problem_blocks(
        torch, device, specification, block_rows
    ):
        requested_terms += requested if kind != "linear" else 0
        stored_terms += stored if kind != "linear" else 0
        if kind == "linear":
            h.copy_(coefficients.to(torch.float32).mul_(0.5))
            constant_quarters.add_(2 * coefficients.sum(dtype=torch.int64))
            linear_nonzero = stored
            continue

        values = coefficients.to(torch.float32)
        if kind == "diagonal":
            symmetric = values + values.T
            factor = -0.25 if representation == "bifurcation" else 0.25
            matrix[start:stop, start:stop].copy_(symmetric * factor)
            h[start:stop].add_(symmetric.sum(dim=1).mul(0.25))
            constant_quarters.add_(coefficients.sum(dtype=torch.int64))
            if row_absolute_sum is not None:
                row_absolute_sum[start:stop].add_(symmetric.abs().sum(dim=1).mul(0.25))
        else:
            factor = -0.25 if representation == "bifurcation" else 0.25
            matrix[start:stop, stop:n].copy_(values * factor)
            matrix[stop:n, start:stop].copy_(values.T * factor)
            h[start:stop].add_(values.sum(dim=1).mul(0.25))
            h[stop:n].add_(values.sum(dim=0).mul(0.25))
            constant_quarters.add_(coefficients.sum(dtype=torch.int64))
            if row_absolute_sum is not None:
                absolute = values.abs()
                row_absolute_sum[start:stop].add_(absolute.sum(dim=1).mul(0.25))
                row_absolute_sum[stop:n].add_(absolute.sum(dim=0).mul(0.25))

    constant = float(constant_quarters.item()) / 4.0
    if representation == "bifurcation":
        matrix[:n, n].copy_(-h)
        matrix[n, :n].copy_(-h)
        matrix[n, n] = 0.0
        scale = None
    else:
        scale = float(torch.max(torch.abs(h) + row_absolute_sum).item())
        scale = scale if scale > 0.0 else 1.0
        matrix.div_(scale)
        h.div_(scale)

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    possible_pairs = n * (n - 1) / 2
    return matrix, h, constant, scale, {
        "requested_quadratic_terms": requested_terms,
        "stored_nonzero_quadratic_terms": stored_terms,
        "stored_nonzero_linear_terms": linear_nonzero,
        "ising_constant": constant,
        "realized_requested_density": requested_terms / possible_pairs if possible_pairs else 0.0,
        "realized_nonzero_density": stored_terms / possible_pairs if possible_pairs else 0.0,
    }


def exact_integer_energies(torch, device, specification, block_rows, samples):
    """Regenerate the seed-defined QUBO and evaluate selected samples exactly in int64."""
    sample_tensor = torch.as_tensor(samples, dtype=torch.bool, device=device).T.contiguous()
    energies = torch.zeros(len(samples), dtype=torch.int64, device=device)
    for kind, start, stop, coefficients, _, _ in _problem_blocks(
        torch, device, specification, block_rows
    ):
        if kind == "linear":
            for index in range(len(samples)):
                energies[index].add_(coefficients[sample_tensor[:, index]].sum(dtype=torch.int64))
        elif kind == "diagonal":
            for index in range(len(samples)):
                selected = sample_tensor[start:stop, index]
                energies[index].add_(coefficients[selected][:, selected].sum(dtype=torch.int64))
        else:
            for index in range(len(samples)):
                rows = sample_tensor[start:stop, index]
                columns = sample_tensor[stop:, index]
                energies[index].add_(coefficients[rows][:, columns].sum(dtype=torch.int64))
    return [int(value) for value in energies.cpu().tolist()]


def run_bifurcation(torch, matrix, agents: int, seed: int, maximum_steps: int, timeout):
    from simulated_bifurcation.optimizer import (
        SimulatedBifurcationEngine,
        SimulatedBifurcationOptimizer,
    )

    torch.random.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    optimizer = SimulatedBifurcationOptimizer(
        agents,
        maximum_steps,
        timeout,
        SimulatedBifurcationEngine.get_engine("ballistic"),
        False,
        False,
        50,
        50,
    )

    # The upstream implementation computes matrix**2 in one expression. Chunking this
    # norm avoids a second 37 GiB allocation for the 100k-variable matrix.
    def chunked_scale(instance, tensor):
        squared_norm = torch.zeros((), dtype=torch.float64, device=tensor.device)
        for start in range(0, tensor.shape[0], 512):
            block = tensor[start : start + 512]
            squared_norm.add_(torch.sum(block * block, dtype=torch.float64))
        instance.quadratic_scale_parameter = 0.5 * math.sqrt(tensor.shape[0] - 1) / torch.sqrt(
            squared_norm
        )

    optimizer._SimulatedBifurcationOptimizer__init_quadratic_scale_parameter = (
        chunked_scale.__get__(optimizer, type(optimizer))
    )
    started = time.perf_counter()
    augmented_spins = optimizer.run_integrator(matrix, early_stopping=True)
    spins = augmented_spins[-1:] * augmented_spins[:-1]
    scores = -0.5 * torch.sum(augmented_spins * (matrix @ augmented_spins), dim=0)
    best = int(torch.argmin(scores).item())
    sample = ((spins[:, best] + 1.0) * 0.5).to(device="cpu", dtype=torch.uint8).tolist()
    torch.cuda.synchronize(matrix.device)
    return sample, time.perf_counter() - started, {
        "seed": seed,
        "agents_on_device": agents,
        "max_steps": maximum_steps,
        "timeout": timeout,
        "mode": "ballistic",
        "heated": False,
        "early_stopping": True,
        "sampling_period": 50,
        "convergence_threshold": 50,
        "completed_steps": int(optimizer.step),
    }


def run_transverse_route(torch, matrix, h, agents, seed, maximum_steps, time_step, interval):
    from solvers.transverse_route import _flow_rhs, _wrap_angles

    torch.random.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    n = matrix.shape[0]
    theta = 2.0 * math.pi * torch.rand((n, agents), device=matrix.device) - math.pi
    best_score = float("inf")
    best_sample = None
    best_step = 0

    def evaluate(step):
        nonlocal best_score, best_sample, best_step
        spins = torch.where(
            torch.cos(theta) >= 0.0, torch.ones_like(theta), -torch.ones_like(theta)
        )
        scores = 0.5 * torch.sum(spins * (matrix @ spins), dim=0) + torch.sum(
            h[:, None] * spins, dim=0
        )
        position = int(torch.argmin(scores).item())
        score = float(scores[position].item())
        if score < best_score or best_sample is None:
            best_score = score
            best_step = step
            best_sample = ((spins[:, position] + 1.0) * 0.5).to(
                device="cpu", dtype=torch.uint8
            ).tolist()

    started = time.perf_counter()
    with torch.no_grad():
        evaluate(0)
        for step in range(1, maximum_steps + 1):
            fraction = (step - 1) / maximum_steps
            kappa = -1.0 + 3.0 * fraction
            slope = _flow_rhs(
                torch,
                matrix,
                h,
                theta,
                kappa=kappa,
                mobility=1.0,
                route_strength=1.0,
                gamma=0.0,
            )
            theta = _wrap_angles(torch, theta + time_step * slope)
            if step % interval == 0 or step == maximum_steps:
                evaluate(step)
        final_mean_abs_y = float(torch.mean(torch.abs(torch.sin(theta))).item())
    torch.cuda.synchronize(matrix.device)
    return best_sample, time.perf_counter() - started, {
        "seed": seed,
        "agents_on_device": agents,
        "max_steps": maximum_steps,
        "time_step": time_step,
        "mobility": 1.0,
        "route_strength": 1.0,
        "gamma": 0.0,
        "kappa_initial": -1.0,
        "kappa_final": 2.0,
        "schedule_exponent": 1.0,
        "integrator": "euler",
        "candidate_interval": interval,
        "best_candidate_step": best_step,
        "final_mean_abs_transverse": final_mean_abs_y,
    }


def device_worker(payload: dict) -> dict:
    import torch

    device_index = payload["device_index"]
    device_position = payload["device_position"]
    device = f"cuda:{device_index}"
    torch.cuda.set_device(device_index)
    output = Path(payload["output"])
    specification = payload["problem"]
    instance_id = payload["instance_id"]
    aliases = ("bifurcation_cuda", "transverse_route_cuda")
    results = {}
    failures = {}
    pending_samples = []
    pending_metadata = []

    for alias in aliases:
        existing = output / "device_runs" / alias / f"cuda_{device_index}.json"
        if existing.is_file():
            results[alias] = json.loads(existing.read_text(encoding="utf-8"))
            (output / "failures" / f"{instance_id}--{alias}--cuda_{device_index}.json").unlink(
                missing_ok=True
            )

    try:
        if "bifurcation_cuda" not in results:
            started = time.perf_counter()
            matrix, _, _, _, generation = materialize_problem(
                torch, device, specification, payload["block_rows"], "bifurcation"
            )
            generation_seconds = time.perf_counter() - started
            agents = agents_for_device(
                payload["bifurcation_agents"], payload["device_count"], device_position,
                payload["agent_mode"],
            )
            sample, solve_seconds, metrics = run_bifurcation(
                torch,
                matrix,
                agents,
                payload["solver_seed"] + device_position,
                payload["bifurcation_max_steps"],
                payload["bifurcation_timeout"],
            )
            del matrix
            torch.cuda.empty_cache()
            pending_samples.append(sample)
            pending_metadata.append(
                ("bifurcation_cuda", solve_seconds, generation_seconds, metrics, generation)
            )
    except Exception as error:
        failures["bifurcation_cuda"] = {
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        torch.cuda.empty_cache()

    try:
        if "transverse_route_cuda" not in results:
            started = time.perf_counter()
            matrix, h, _, scale, generation = materialize_problem(
                torch, device, specification, payload["block_rows"], "transverse_route"
            )
            generation_seconds = time.perf_counter() - started
            agents = agents_for_device(
                payload["transverse_route_agents"], payload["device_count"], device_position,
                payload["agent_mode"],
            )
            sample, solve_seconds, metrics = run_transverse_route(
                torch,
                matrix,
                h,
                agents,
                payload["solver_seed"] + device_position,
                payload["transverse_route_max_steps"],
                payload["transverse_route_time_step"],
                payload["transverse_route_candidate_interval"],
            )
            metrics["interaction_scale"] = scale
            del matrix, h
            torch.cuda.empty_cache()
            pending_samples.append(sample)
            pending_metadata.append(
                ("transverse_route_cuda", solve_seconds, generation_seconds, metrics, generation)
            )
    except Exception as error:
        failures["transverse_route_cuda"] = {
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        torch.cuda.empty_cache()

    if pending_samples:
        energies = exact_integer_energies(
            torch, device, specification, payload["block_rows"], pending_samples
        )
        device_name = torch.cuda.get_device_name(device_index)
        for sample, energy, metadata in zip(pending_samples, energies, pending_metadata):
            alias, solve_seconds, generation_seconds, metrics, generation = metadata
            result = {
                "schema_version": 2,
                "run_id": f"{alias}-cuda{device_index}-{payload['benchmark_id'][:12]}",
                "benchmark_solver_alias": alias,
                "problem": {
                    "instance_id": instance_id,
                    "problem_type": "implicit_random_qubo",
                    "num_variables": specification["num_variables"],
                    "quadratic_terms": generation["stored_nonzero_quadratic_terms"],
                    "compact_manifest": "problem_manifest.json",
                },
                "solver": {
                    "type": (
                        "simulated_bifurcation"
                        if alias == "bifurcation_cuda"
                        else "transverse_route"
                    ),
                    "name": (
                        "Simulated Bifurcation"
                        if alias == "bifurcation_cuda"
                        else "Transverse-Route Geometric Flow"
                    ),
                    "version": (
                        f"simulated-bifurcation {package_version('simulated-bifurcation')}"
                        if alias == "bifurcation_cuda"
                        else f"built in (PyTorch {torch.__version__})"
                    ),
                    "parameters": metrics,
                    "device": f"cuda:{device_index} ({device_name})",
                },
                "status": "completed",
                "solution": {"sample": sample, "energy": energy},
                "timing": {
                    "wall_seconds": solve_seconds,
                    "problem_materialization_seconds": generation_seconds,
                },
                "verification": {
                    "binary_sample": True,
                    "method": "deterministic coefficient regeneration with int64 accumulation",
                    "reported_energy": energy,
                    "recomputed_energy": energy,
                    "absolute_energy_difference": 0,
                    "passed": True,
                },
                "solver_metrics": {**metrics, **generation, "optimality_proven": False},
            }
            result_path = output / "device_runs" / alias / f"cuda_{device_index}.json"
            atomic_json(result_path, result)
            (output / "failures" / f"{instance_id}--{alias}--cuda_{device_index}.json").unlink(
                missing_ok=True
            )
            results[alias] = result

    for alias, failure in failures.items():
        atomic_json(
            output / "failures" / f"{instance_id}--{alias}--cuda_{device_index}.json",
            {
                "instance_id": instance_id,
                "solver_alias": alias,
                "solver_type": (
                    "simulated_bifurcation"
                    if alias == "bifurcation_cuda"
                    else "transverse_route"
                ),
                "parameters": {
                    "agents": payload[
                        "bifurcation_agents"
                        if alias == "bifurcation_cuda"
                        else "transverse_route_agents"
                    ]
                },
                "device": device,
                "occurred_at": utc_now(),
                **failure,
            },
        )
    return {
        "device_index": device_index,
        "completed": sorted(results),
        "failures": failures,
    }


def build_reports(output: Path, config: dict, instance_id: str, device_indices: list[int]) -> None:
    solver_definitions = config["solvers"]
    aggregate_results = {}
    device_counts = {}
    generation_signatures = set()
    rows = []
    for alias, definition in solver_definitions.items():
        device_results = []
        for device in device_indices:
            path = output / "device_runs" / alias / f"cuda_{device}.json"
            if path.is_file():
                device_result = json.loads(path.read_text(encoding="utf-8"))
                device_results.append(device_result)
                metrics = device_result["solver_metrics"]
                generation_signatures.add(
                    (
                        metrics["requested_quadratic_terms"],
                        metrics["stored_nonzero_quadratic_terms"],
                        metrics["stored_nonzero_linear_terms"],
                        metrics["ising_constant"],
                    )
                )
        device_counts[alias] = len(device_results)
        if not device_results:
            continue
        best = min(device_results, key=lambda result: result["solution"]["energy"])
        combined = {
            **best,
            "run_id": f"{alias}-{len(device_indices)}gpu-{config['benchmark_id'][:12]}",
            "benchmark_solver_alias": alias,
            "solver": {**best["solver"], "device": f"{len(device_results)} CUDA devices"},
            "timing": {
                "wall_seconds": max(result["timing"]["wall_seconds"] for result in device_results),
                "aggregate_device_seconds": sum(
                    result["timing"]["wall_seconds"] for result in device_results
                ),
            },
            "solver_metrics": {
                **best["solver_metrics"],
                "devices_requested": len(device_indices),
                "devices_completed": len(device_results),
                "best_device": best["solver"]["device"],
                "device_runs": [
                    {
                        "device": result["solver"]["device"],
                        "energy": result["solution"]["energy"],
                        "wall_seconds": result["timing"]["wall_seconds"],
                        "materialization_seconds": result["timing"][
                            "problem_materialization_seconds"
                        ],
                    }
                    for result in device_results
                ],
            },
        }
        raw_path = output / "raw" / instance_id / f"{alias}.json"
        atomic_json(raw_path, combined)
        aggregate_results[alias] = combined

    if len(generation_signatures) > 1:
        raise RuntimeError(
            "Deterministic problem-generation signatures differ across solver/device runs"
        )

    energies = {alias: result["solution"]["energy"] for alias, result in aggregate_results.items()}
    best_energy = min(energies.values()) if energies else None
    best_aliases = [alias for alias, energy in energies.items() if energy == best_energy]
    samples = {alias: result["solution"]["sample"] for alias, result in aggregate_results.items()}
    for alias, definition in solver_definitions.items():
        if alias not in aggregate_results:
            rows.append(
                {
                    "solver_alias": alias,
                    "solver": definition["label"],
                    "completed": False,
                    "device_runs": 0,
                }
            )
            continue
        result = aggregate_results[alias]
        energy = energies[alias]
        distance = min(
            sum(left != right for left, right in zip(samples[alias], samples[winner]))
            for winner in best_aliases
        )
        rows.append(
            {
                "solver_alias": alias,
                "solver": definition["label"],
                "completed": True,
                "device_runs": device_counts[alias],
                "status": "completed",
                "energy": energy,
                "rank": 1 + sum(other < energy for other in energies.values()),
                "is_best_or_tied": alias in best_aliases,
                "absolute_gap_to_best": energy - best_energy,
                "relative_gap_to_best": (energy - best_energy) / max(1.0, abs(best_energy)),
                "hamming_to_nearest_best": distance,
                "hamming_fraction_to_nearest_best": distance / config["problem"]["num_variables"],
                "sample_ones": sum(samples[alias]),
                "wall_seconds": result["timing"]["wall_seconds"],
                "device": result["solver"]["device"],
                "verification_passed": True,
                "result_path": str((output / "raw" / instance_id / f"{alias}.json").resolve()),
            }
        )

    pairwise_hamming = {alias: {} for alias in solver_definitions}
    pairwise_energy = {alias: {} for alias in solver_definitions}
    for left in solver_definitions:
        for right in solver_definitions:
            if left in samples and right in samples:
                distance = sum(a != b for a, b in zip(samples[left], samples[right]))
                pairwise_hamming[left][right] = {
                    "distance": distance,
                    "fraction": distance / config["problem"]["num_variables"],
                }
                pairwise_energy[left][right] = energies[left] - energies[right]
            else:
                pairwise_hamming[left][right] = None
                pairwise_energy[left][right] = None

    report = {
        "schema_version": 2,
        "instance_id": instance_id,
        "problem": {
            "instance_id": instance_id,
            "problem_type": "implicit_random_qubo",
            "num_variables": config["problem"]["num_variables"],
            "quadratic_terms": next(
                (result["problem"]["quadratic_terms"] for result in aggregate_results.values()),
                round(
                    config["problem"]["quadratic_density"]
                    * config["problem"]["num_variables"]
                    * (config["problem"]["num_variables"] - 1)
                    / 2
                ),
            ),
        },
        "complete_comparison": (
            len(aggregate_results) == len(solver_definitions)
            and all(count == len(device_indices) for count in device_counts.values())
        ),
        "best_energy": best_energy,
        "best_solver_aliases": best_aliases,
        "solver_results": rows,
        "pairwise_hamming": pairwise_hamming,
        "pairwise_energy_difference_left_minus_right": pairwise_energy,
    }
    atomic_json(output / "instances" / f"{instance_id}.json", report)
    write_csv(
        output / "instances" / f"{instance_id}.csv",
        rows,
        [
            "solver_alias", "solver", "completed", "device_runs", "status", "energy", "rank",
            "is_best_or_tied", "absolute_gap_to_best", "relative_gap_to_best",
            "hamming_to_nearest_best", "hamming_fraction_to_nearest_best", "sample_ones",
            "wall_seconds", "device", "verification_passed", "result_path",
        ],
    )
    markdown = [
        f"# {len(device_indices)}-GPU benchmark: {instance_id}",
        "",
        f"- Complete comparison: {'yes' if report['complete_comparison'] else 'no'}",
        f"- Best energy: {best_energy}",
        f"- Best solver: {', '.join(best_aliases)}",
        "",
        "| Solver | Device runs | Energy | Rank | Gap | Wall s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["completed"]:
            markdown.append(
                f"| {row['solver']} | {row['device_runs']} | {row['energy']} | {row['rank']} | "
                f"{row['absolute_gap_to_best']} | {row['wall_seconds']:.3f} |"
            )
        else:
            markdown.append(f"| {row['solver']} | 0 | — | — | — | — |")
    atomic_text(output / "instances" / f"{instance_id}.md", "\n".join(markdown) + "\n")

    summary = []
    for row in rows:
        summary.append(
            {
                "solver_alias": row["solver_alias"],
                "solver": row["solver"],
                "completed_runs": int(row["completed"]),
                "failed_runs": int(not row["completed"]),
                "wins_or_ties": int(row.get("is_best_or_tied", False)),
                "mean_rank": row.get("rank"),
                "mean_energy": row.get("energy"),
                "mean_absolute_gap_to_best": row.get("absolute_gap_to_best"),
                "mean_relative_gap_to_best": row.get("relative_gap_to_best"),
                "mean_wall_seconds": row.get("wall_seconds"),
            }
        )
    aggregate = {
        "schema_version": 2,
        "instances_requested": 1,
        "complete_comparisons": int(report["complete_comparison"]),
        "solver_summary": summary,
    }
    atomic_json(output / "aggregate_report.json", aggregate)
    write_csv(output / "aggregate_report.csv", summary, list(summary[0]))
    write_csv(output / "all_solver_results.csv", rows, list(rows[0]))
    atomic_text(output / "aggregate_report.md", "\n".join(markdown) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Prepare an implicit Random QUBO and benchmark SB versus TRF on 8 GPUs."
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--devices", help="Comma-separated CUDA indices (default: 0 through 7)")
    result.add_argument("--expected-devices", type=int, default=8)
    result.add_argument("--num-variables", type=int, default=100_000)
    result.add_argument("--quadratic-density", type=float, default=0.25)
    result.add_argument("--problem-seed", type=int, default=0)
    result.add_argument("--solver-seed", type=int, default=0)
    result.add_argument("--block-rows", type=int, default=512)
    result.add_argument("--agent-mode", choices=("shard", "replicate"), default="shard")
    result.add_argument("--bifurcation-agents", type=int, default=128)
    result.add_argument("--bifurcation-max-steps", type=int, default=10_000)
    result.add_argument("--bifurcation-timeout", type=float, default=60.0)
    result.add_argument("--transverse-route-agents", type=int, default=64)
    result.add_argument("--transverse-route-max-steps", type=int, default=1_000)
    result.add_argument("--transverse-route-time-step", type=float, default=0.05)
    result.add_argument("--transverse-route-candidate-interval", type=int, default=25)
    result.add_argument("--minimum-free-gib", type=float, default=45.0)
    result.add_argument("--skip-memory-check", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.num_variables < 1 or not 0.0 <= args.quadratic_density <= 1.0:
        raise ValueError("num_variables must be positive and quadratic_density must be in [0, 1]")
    if args.expected_devices < 1 or args.block_rows < 1:
        raise ValueError("expected_devices and block_rows must be positive")
    if min(
        args.bifurcation_agents,
        args.bifurcation_max_steps,
        args.transverse_route_agents,
        args.transverse_route_max_steps,
        args.transverse_route_candidate_interval,
    ) < 1:
        raise ValueError("agent, step, and candidate-interval counts must be positive")
    if args.bifurcation_timeout < 0 or args.transverse_route_time_step <= 0:
        raise ValueError("timeout must be nonnegative and time_step must be positive")
    if args.minimum_free_gib < 0:
        raise ValueError("minimum_free_gib must be nonnegative")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation is required")
    devices = parse_devices(args.devices, args.expected_devices, torch.cuda.device_count())
    matrix_gib = (args.num_variables + 1) ** 2 * 4 / 1024**3
    if not args.skip_memory_check:
        insufficient = []
        for device in devices:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes / 1024**3 < args.minimum_free_gib:
                insufficient.append((device, free_bytes / 1024**3))
        if insufficient:
            details = ", ".join(
                f"cuda:{device}={free:.1f} GiB free" for device, free in insufficient
            )
            raise RuntimeError(
                f"Insufficient free GPU memory ({details}); {args.minimum_free_gib:.1f} GiB "
                "per device is required by default."
            )

    specification = {
        "schema_version": 1,
        "representation": "implicit_seeded_random_qubo",
        "num_variables": args.num_variables,
        "quadratic_density": args.quadratic_density,
        "distribution": "integer",
        "coefficient_min": -10,
        "coefficient_max": 10,
        "normal_scale": 5.0,
        "zero_linear_probability": 0.0,
        "offset": 0.0,
        "seed": args.problem_seed,
        "block_rows": args.block_rows,
        "generation": "deterministic blockwise PyTorch CUDA RNG",
    }
    instance_id = problem_id(specification)
    solvers = {
        "bifurcation_cuda": {
            "solver_type": "simulated_bifurcation",
            "label": f"Simulated Bifurcation ({args.expected_devices}× CUDA)",
            "parameters": {
                "agents": args.bifurcation_agents,
                "max_steps": args.bifurcation_max_steps,
                "timeout": args.bifurcation_timeout,
                "mode": "ballistic",
                "heated": False,
                "early_stopping": True,
                "sampling_period": 50,
                "convergence_threshold": 50,
                "dtype": "float32",
                "seed": args.solver_seed,
                "agent_distribution": args.agent_mode,
                "max_variables": args.num_variables,
                "num_snapshots": 0,
            },
        },
        "transverse_route_cuda": {
            "solver_type": "transverse_route",
            "label": f"Transverse-Route Flow ({args.expected_devices}× CUDA)",
            "parameters": {
                "agents": args.transverse_route_agents,
                "max_steps": args.transverse_route_max_steps,
                "time_step": args.transverse_route_time_step,
                "mobility": 1.0,
                "route_strength": 1.0,
                "gamma": 0.0,
                "kappa_initial": -1.0,
                "kappa_final": 2.0,
                "schedule_exponent": 1.0,
                "integrator": "euler",
                "candidate_interval": args.transverse_route_candidate_interval,
                "matrix_format": "dense",
                "dtype": "float32",
                "seed": args.solver_seed,
                "agent_distribution": args.agent_mode,
                "max_variables": args.num_variables,
                "max_dense_variables": args.num_variables,
                "num_snapshots": 0,
            },
        },
    }
    identity_source = {"problem": specification, "solvers": solvers, "devices": devices}
    benchmark_id = hashlib.sha256(
        json.dumps(identity_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    print(f"Problem: {instance_id}")
    print(f"Variables: {args.num_variables:,}")
    expected_terms = (
        args.quadratic_density * args.num_variables * (args.num_variables - 1) / 2
    )
    print(f"Expected upper-triangular terms: {expected_terms:,.0f}")
    print(f"Dense FP32 matrix per device: {matrix_gib:.2f} GiB")
    print(f"CUDA devices: {', '.join(f'cuda:{item}' for item in devices)}")
    print(f"Output: {args.output.resolve()}")
    if args.dry_run:
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "failures").mkdir(parents=True, exist_ok=True)
    config_path = output / "benchmark_config.json"
    previous_config = (
        json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    )
    if previous_config and previous_config.get("benchmark_id") != benchmark_id:
        raise RuntimeError(
            f"Output {output} contains a different benchmark. Select a new --output path "
            "instead of mixing incompatible device results."
        )
    config = {
        "schema_version": 2,
        "benchmark_type": f"implicit_random_qubo_{len(devices)}gpu",
        "benchmark_id": benchmark_id,
        "created_at": (
            previous_config.get("created_at", utc_now()) if previous_config else utc_now()
        ),
        "problem": specification,
        "problem_manifest": "problem_manifest.json",
        "solvers": solvers,
        "devices": devices,
        "notes": [
            f"The {expected_terms:,.0f}-selected-pair QUBO uses a compact seeded manifest.",
            "Both solvers receive independently materialized copies of the identical matrix.",
            f"Default agent batches are {args.agent_mode}ed across the CUDA devices.",
            "Device workers use deterministic substreams derived from the default solver seed.",
            "Allocation safety ceilings are raised; algorithmic settings retain adapter defaults.",
            "Final objectives are verified by deterministic regeneration with int64 accumulation.",
        ],
    }
    atomic_json(output / "problem_manifest.json", specification)
    atomic_json(config_path, config)
    atomic_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "torch", "simulated-bifurcation")
            },
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(device) for device in devices],
            "errors": [],
        },
    )

    progress_path = output / "progress.json"
    previous_progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else None
    )
    if previous_progress and previous_progress.get("benchmark_id") != benchmark_id:
        previous_progress = None
    previous_tasks = previous_progress.get("tasks", {}) if previous_progress else {}
    tasks = {}
    for device in devices:
        for alias in solvers:
            result_path = output / "device_runs" / alias / f"cuda_{device}.json"
            task_key = f"{instance_id}::{alias}::cuda_{device}"
            previous_task = previous_tasks.get(task_key, {})
            tasks[task_key] = {
                "instance_id": instance_id,
                "solver_alias": alias,
                "device": f"cuda:{device}",
                "status": "completed" if result_path.is_file() else "pending",
                "attempts": max(int(result_path.is_file()), previous_task.get("attempts", 0)),
                "elapsed_seconds": None,
                "result_path": str(result_path),
                "error": None,
                "started_at": previous_task.get("started_at"),
                "finished_at": previous_task.get("finished_at"),
            }
    progress = {
        "schema_version": 2,
        "benchmark_id": benchmark_id,
        "created_at": (
            previous_progress.get("created_at", utc_now())
            if previous_progress
            else utc_now()
        ),
        "updated_at": utc_now(),
        "tasks": tasks,
    }
    atomic_json(progress_path, progress)

    payloads = []
    for position, device in enumerate(devices):
        missing = any(
            progress["tasks"][f"{instance_id}::{alias}::cuda_{device}"]["status"] != "completed"
            for alias in solvers
        )
        if not missing:
            continue
        payloads.append(
            {
                **vars(args),
                "output": str(output),
                "problem": specification,
                "instance_id": instance_id,
                "benchmark_id": benchmark_id,
                "device_index": device,
                "device_position": position,
                "device_count": len(devices),
            }
        )

    started_at = utc_now()
    for payload in payloads:
        device = payload["device_index"]
        for alias in solvers:
            task = progress["tasks"][f"{instance_id}::{alias}::cuda_{device}"]
            if task["status"] != "completed":
                task["status"] = "running"
                task["attempts"] += 1
                task["started_at"] = started_at
    progress["updated_at"] = started_at
    atomic_json(progress_path, progress)

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(devices), mp_context=context
    ) as executor:
        future_by_device = {
            executor.submit(device_worker, payload): payload["device_index"]
            for payload in payloads
        }
        for future in concurrent.futures.as_completed(future_by_device):
            device = future_by_device[future]
            try:
                outcome = future.result()
                print(
                    f"cuda:{device}: completed={','.join(outcome['completed']) or 'none'} "
                    f"failed={','.join(outcome['failures']) or 'none'}",
                    flush=True,
                )
            except Exception as error:
                print(f"cuda:{device}: worker failed: {error}", file=sys.stderr, flush=True)
                for alias in solvers:
                    result_path = output / "device_runs" / alias / f"cuda_{device}.json"
                    if result_path.is_file():
                        continue
                    atomic_json(
                        output / "failures" / f"{instance_id}--{alias}--cuda_{device}.json",
                        {
                            "instance_id": instance_id,
                            "solver_alias": alias,
                            "solver_type": solvers[alias]["solver_type"],
                            "parameters": solvers[alias]["parameters"],
                            "device": f"cuda:{device}",
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                            "occurred_at": utc_now(),
                        },
                    )

    completed = failed = 0
    for key, task in progress["tasks"].items():
        result_path = Path(task["result_path"])
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            failure_name = (
                f"{instance_id}--{task['solver_alias']}--"
                f"{task['device'].replace(':', '_')}.json"
            )
            (output / "failures" / failure_name).unlink(missing_ok=True)
            task.update(
                {
                    "status": "completed",
                    "attempts": max(1, task["attempts"]),
                    "elapsed_seconds": result["timing"]["wall_seconds"],
                    "finished_at": utc_now(),
                }
            )
            completed += 1
        else:
            task["status"] = "failed"
            task["attempts"] = max(1, task["attempts"])
            task["finished_at"] = utc_now()
            task["error"] = "No result file was produced; inspect failures/"
            failed += 1
    progress["updated_at"] = utc_now()
    atomic_json(progress_path, progress)
    build_reports(output, config, instance_id, devices)
    print(f"Finished: {completed}/{len(tasks)} device runs completed; {failed} failed")
    print(f"Aggregate report: {(output / 'aggregate_report.md').resolve()}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
