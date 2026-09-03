"""Batched transverse-route geometric-flow solver for QUBO problems."""

from __future__ import annotations

import math
import time

import numpy as np

from problems.common import qubo_energy

from .common import (
    SolverCapabilityError,
    equally_spaced_targets,
    package_version,
    require_package,
    snapshot_record,
)


NAME = "Transverse-Route Geometric Flow"
DESCRIPTION = """
Evolves batches of Ising spins on circles using the first-order geometric flow from
docs/new_solver.md. Longitudinal and endpoint-oriented route fields use the same Ising
coupling matrix; an optional higher-order route channel remains active near mid-route.
"""

PARAMETERS = {
    "agents": {
        "type": "integer",
        "default": 64,
        "min": 1,
        "description": "Independent angular trajectories evolved in parallel",
    },
    "max_steps": {
        "type": "integer",
        "default": 1_000,
        "min": 1,
        "description": "Number of fixed-step geometric-flow updates",
    },
    "time_step": {
        "type": "number",
        "default": 0.05,
        "min": 1e-12,
        "description": "Dimensionless integration step after interaction normalization",
    },
    "mobility": {
        "type": "number",
        "default": 1.0,
        "min": 1e-12,
        "description": "Positive mobility multiplying the negative energy gradient",
    },
    "route_strength": {
        "type": "number",
        "default": 1.0,
        "min": 0.0,
        "description": "Weight of the r=x*y channel; one preserves exact flip curvature",
    },
    "gamma": {
        "type": "number",
        "default": 0.0,
        "min": 0.0,
        "description": "Weight of the optional p=y^3 route-separation channel",
    },
    "kappa_initial": {
        "type": "number",
        "default": -1.0,
        "description": "Initial transverse confinement (negative opens the manifold)",
    },
    "kappa_final": {
        "type": "number",
        "default": 2.0,
        "description": "Final transverse confinement (positive favors binary endpoints)",
    },
    "schedule_exponent": {
        "type": "number",
        "default": 1.0,
        "min": 1e-12,
        "description": "Power-law exponent for the monotone confinement schedule",
    },
    "integrator": {
        "type": "choice",
        "default": "euler",
        "choices": ["euler", "heun"],
        "description": "Explicit Euler or second-order Heun integration",
    },
    "candidate_interval": {
        "type": "integer",
        "default": 25,
        "min": 1,
        "description": "Flow steps between exact discrete candidate checks",
    },
    "matrix_format": {
        "type": "choice",
        "default": "auto",
        "choices": ["auto", "dense", "sparse"],
        "description": "Dense GEMM, sparse SpMM, or density-based automatic selection",
    },
    "sparse_threshold": {
        "type": "number",
        "default": 0.15,
        "min": 0.0,
        "max": 1.0,
        "description": "Maximum directed matrix density selected as sparse in auto mode",
    },
    "dtype": {
        "type": "choice",
        "default": "float32",
        "choices": ["float32", "float64"],
        "description": "PyTorch trajectory and interaction precision",
    },
    "device": {
        "type": "string",
        "default": "cpu",
        "description": "cpu, auto, cuda, or a CUDA index such as cuda:0",
    },
    "seed": {
        "type": "integer",
        "default": 0,
        "min": 0,
        "description": "PyTorch seed for uniformly random initial angles",
    },
    "max_variables": {
        "type": "integer",
        "default": 100_000,
        "min": 1,
        "description": "Safety limit for trajectory state allocation",
    },
    "max_dense_variables": {
        "type": "integer",
        "default": 10_000,
        "min": 1,
        "description": "Safety limit for an explicitly or automatically dense matrix",
    },
    "num_snapshots": {
        "type": "integer",
        "default": 0,
        "min": 0,
        "description": "Equally spaced best-seen flow checkpoints; zero disables snapshots",
    },
}


def version() -> str:
    return f"built in (PyTorch {package_version('torch')})"


def _qubo_to_ising(qubo: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return h, edge endpoints, edge J values, and the binary-to-Ising constant.

    The returned off-diagonal edge values are the entries of symmetric ``J`` in
    ``0.5*s.T@J@s + h.T@s + constant``. Duplicate input terms are supported.
    """
    num_variables = int(qubo["num_variables"])
    h = np.zeros(num_variables, dtype=np.float64)
    constant = float(qubo.get("offset", 0.0))
    for variable, coefficient in qubo["linear"]:
        value = float(coefficient)
        h[int(variable)] += value / 2.0
        constant += value / 2.0

    edge_values: dict[tuple[int, int], float] = {}
    for first, second, coefficient in qubo["quadratic"]:
        first = int(first)
        second = int(second)
        value = float(coefficient)
        if first == second:
            # Be defensive about non-canonical QUBOs: z_i^2 == z_i.
            h[first] += value / 2.0
            constant += value / 2.0
            continue
        if first > second:
            first, second = second, first
        coupling = value / 4.0
        edge_values[(first, second)] = edge_values.get((first, second), 0.0) + coupling
        h[first] += coupling
        h[second] += coupling
        constant += coupling

    nonzero = [(edge, value) for edge, value in edge_values.items() if value != 0.0]
    first = np.asarray([edge[0] for edge, _ in nonzero], dtype=np.int64)
    second = np.asarray([edge[1] for edge, _ in nonzero], dtype=np.int64)
    values = np.asarray([value for _, value in nonzero], dtype=np.float64)
    return h, first, second, values, constant


def _interaction_scale(
    h: np.ndarray, first: np.ndarray, second: np.ndarray, values: np.ndarray
) -> float:
    row_bounds = np.abs(h).copy()
    if values.size:
        magnitudes = np.abs(values)
        np.add.at(row_bounds, first, magnitudes)
        np.add.at(row_bounds, second, magnitudes)
    scale = float(row_bounds.max(initial=0.0))
    return scale if scale > 0.0 else 1.0


def _resolve_device(torch, requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu" or requested == "cuda" or requested.startswith("cuda:"):
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SolverCapabilityError(
                "CUDA was requested for transverse-route flow, but this PyTorch "
                "installation cannot access a CUDA GPU. Use device=cpu or install a "
                "CUDA-enabled PyTorch build."
            )
        try:
            torch.empty(0, device=requested)
        except (RuntimeError, ValueError) as error:
            raise SolverCapabilityError(f"Cannot use PyTorch device {requested!r}: {error}") from error
        return requested
    raise ValueError("device must be 'cpu', 'auto', 'cuda', or a CUDA index such as 'cuda:0'")


def _build_matrix(torch, num_variables, first, second, values, dtype, device, matrix_format):
    if matrix_format == "dense":
        matrix = torch.zeros((num_variables, num_variables), dtype=dtype, device=device)
        if values.size:
            rows = torch.as_tensor(np.concatenate((first, second)), device=device)
            columns = torch.as_tensor(np.concatenate((second, first)), device=device)
            data = torch.as_tensor(np.concatenate((values, values)), dtype=dtype, device=device)
            matrix.index_put_((rows, columns), data, accumulate=True)
        return matrix

    if values.size:
        indices = torch.as_tensor(
            np.stack((np.concatenate((first, second)), np.concatenate((second, first)))),
            dtype=torch.int64,
            device=device,
        )
        data = torch.as_tensor(np.concatenate((values, values)), dtype=dtype, device=device)
    else:
        indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        data = torch.empty(0, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices, data, (num_variables, num_variables), dtype=dtype, device=device
    ).coalesce()


def _matmul(torch, matrix, features):
    return torch.sparse.mm(matrix, features) if matrix.is_sparse else matrix @ features


def _flow_rhs(
    torch,
    matrix,
    h,
    theta,
    *,
    kappa: float,
    mobility: float,
    route_strength: float,
    gamma: float,
):
    """Evaluate Eq. (4), batching all active J products into one operation."""
    x = torch.cos(theta)
    y = torch.sin(theta)
    route = x * y
    features = [x]
    if route_strength != 0.0:
        features.append(route)
    if gamma != 0.0:
        features.append(y * y * y)
    fields = _matmul(torch, matrix, torch.cat(features, dim=1))

    agents = theta.shape[1]
    offset = agents
    longitudinal_field = fields[:, :agents] + h[:, None]
    flow = y * longitudinal_field
    if route_strength != 0.0:
        route_field = fields[:, offset : offset + agents]
        offset += agents
        flow = flow - route_strength * (x * x - y * y) * route_field
    if gamma != 0.0:
        higher_field = fields[:, offset : offset + agents]
        flow = flow - 3.0 * gamma * x * y * y * higher_field
    return mobility * (flow - kappa * x * y)


def _wrap_angles(torch, theta):
    return torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi


def _qubo_arrays(qubo: dict):
    num_variables = int(qubo["num_variables"])
    linear = np.zeros(num_variables, dtype=np.float64)
    for variable, coefficient in qubo["linear"]:
        linear[int(variable)] += float(coefficient)
    first = np.asarray([term[0] for term in qubo["quadratic"]], dtype=np.int64)
    second = np.asarray([term[1] for term in qubo["quadratic"]], dtype=np.int64)
    values = np.asarray([term[2] for term in qubo["quadratic"]], dtype=np.float64)
    return linear, first, second, values, float(qubo.get("offset", 0.0))


def _batch_qubo_energies(samples, linear, first, second, values, offset):
    """Evaluate original QUBO coefficients in FP64 without a large edge temporary."""
    energies = offset + linear @ samples
    for start in range(0, len(values), 8_192):
        stop = min(start + 8_192, len(values))
        products = samples[first[start:stop]] * samples[second[start:stop]]
        energies += values[start:stop] @ products
    return energies


def solve(qubo: dict, parameters: dict) -> dict:
    started = time.perf_counter()
    require_package("torch", "torch", NAME)
    import torch

    finite_parameters = (
        "time_step",
        "mobility",
        "route_strength",
        "gamma",
        "kappa_initial",
        "kappa_final",
        "schedule_exponent",
        "sparse_threshold",
    )
    for name in finite_parameters:
        if not math.isfinite(float(parameters[name])):
            raise ValueError(f"{name} must be finite")

    num_variables = int(qubo["num_variables"])
    if num_variables > parameters["max_variables"]:
        raise SolverCapabilityError(
            f"Transverse-route flow is limited to {parameters['max_variables']:,} variables "
            f"by max_variables, but this QUBO has {num_variables:,}."
        )
    if parameters["kappa_final"] < parameters["kappa_initial"]:
        raise ValueError("kappa_final must be at least kappa_initial for a monotone schedule")

    device = _resolve_device(torch, parameters["device"])
    torch_dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64
    element_bytes = 4 if parameters["dtype"] == "float32" else 8
    h_numpy, first, second, edge_values, ising_constant = _qubo_to_ising(qubo)
    scale = _interaction_scale(h_numpy, first, second, edge_values)
    h_numpy /= scale
    edge_values /= scale

    directed_nonzeros = 2 * len(edge_values)
    density = directed_nonzeros / max(1, num_variables * num_variables)
    matrix_format = parameters["matrix_format"]
    if matrix_format == "auto":
        matrix_format = "sparse" if density <= parameters["sparse_threshold"] else "dense"
    if matrix_format == "dense" and num_variables > parameters["max_dense_variables"]:
        estimated_mib = num_variables * num_variables * element_bytes / 1024**2
        raise SolverCapabilityError(
            f"Dense transverse-route flow needs a {num_variables:,} x {num_variables:,} "
            f"matrix (about {estimated_mib:,.1f} MiB before working memory), above "
            f"max_dense_variables={parameters['max_dense_variables']:,}. Use matrix_format=sparse."
        )

    matrix = _build_matrix(
        torch,
        num_variables,
        first,
        second,
        edge_values,
        torch_dtype,
        device,
        matrix_format,
    )
    h = torch.as_tensor(h_numpy, dtype=torch_dtype, device=device)
    qubo_linear, qubo_first, qubo_second, qubo_values, qubo_offset = _qubo_arrays(qubo)
    agents = parameters["agents"]
    max_steps = parameters["max_steps"]
    snapshot_targets = equally_spaced_targets(max_steps, parameters["num_snapshots"])
    snapshot_target_set = set(snapshot_targets)
    snapshots = []

    torch.manual_seed(parameters["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(parameters["seed"])
    theta = 2.0 * math.pi * torch.rand(
        (num_variables, agents), dtype=torch_dtype, device=device
    ) - math.pi

    best_energy = float("inf")
    best_sample: list[int] | None = None
    best_step = 0
    candidate_batches = 0

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))

    def evaluate_candidates(step: int) -> None:
        nonlocal best_energy, best_sample, best_step, candidate_batches
        x = torch.cos(theta)
        samples = (x >= 0.0).to(device="cpu", dtype=torch.uint8).numpy()
        energies = _batch_qubo_energies(
            samples, qubo_linear, qubo_first, qubo_second, qubo_values, qubo_offset
        )
        position = int(np.argmin(energies))
        energy = float(energies[position])
        candidate_batches += 1
        if energy < best_energy or best_sample is None:
            best_energy = energy
            best_step = step
            best_sample = samples[:, position].tolist()

    def kappa_at(fraction: float) -> float:
        return parameters["kappa_initial"] + (
            parameters["kappa_final"] - parameters["kappa_initial"]
        ) * fraction ** parameters["schedule_exponent"]

    with torch.no_grad():
        evaluate_candidates(0)
        for step in range(1, max_steps + 1):
            start_fraction = (step - 1) / max_steps
            start_kappa = kappa_at(start_fraction)
            first_slope = _flow_rhs(
                torch,
                matrix,
                h,
                theta,
                kappa=start_kappa,
                mobility=parameters["mobility"],
                route_strength=parameters["route_strength"],
                gamma=parameters["gamma"],
            )
            if parameters["integrator"] == "heun":
                predictor = _wrap_angles(torch, theta + parameters["time_step"] * first_slope)
                second_slope = _flow_rhs(
                    torch,
                    matrix,
                    h,
                    predictor,
                    kappa=kappa_at(step / max_steps),
                    mobility=parameters["mobility"],
                    route_strength=parameters["route_strength"],
                    gamma=parameters["gamma"],
                )
                theta = _wrap_angles(
                    torch,
                    theta + 0.5 * parameters["time_step"] * (first_slope + second_slope),
                )
            else:
                theta = _wrap_angles(torch, theta + parameters["time_step"] * first_slope)

            should_evaluate = (
                step % parameters["candidate_interval"] == 0
                or step == max_steps
                or step in snapshot_target_set
            )
            if should_evaluate:
                evaluate_candidates(step)

            if step in snapshot_target_set:
                synchronize()
                snapshots.append(
                    snapshot_record(
                        index=len(snapshots) + 1,
                        count=len(snapshot_targets),
                        unit="flow_steps",
                        target=step,
                        total=max_steps,
                        sample=best_sample,
                        reported_energy=qubo_energy(qubo, best_sample),
                        elapsed_seconds=time.perf_counter() - started,
                        metrics={
                            "kappa": kappa_at(step / max_steps),
                            "best_candidate_step": best_step,
                        },
                    )
                )

        final_y = torch.sin(theta)
        mean_abs_y = float(torch.mean(torch.abs(final_y)).item())
        max_abs_y = float(torch.max(torch.abs(final_y)).item())

    synchronize()
    assert best_sample is not None
    reported_energy = qubo_energy(qubo, best_sample)
    if matrix_format == "dense":
        matrix_mib = num_variables * num_variables * element_bytes / 1024**2
    else:
        matrix_mib = directed_nonzeros * (2 * 8 + element_bytes) / 1024**2
    device_name = "cpu"
    if device.startswith("cuda"):
        device_name = f"{device} ({torch.cuda.get_device_name(torch.device(device))})"

    return {
        "status": "completed",
        "sample": best_sample,
        "reported_energy": reported_energy,
        "device": device_name,
        "metrics": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "matrix_format": matrix_format,
            "matrix_density": density,
            "matrix_storage_mib": matrix_mib,
            "interaction_scale": scale,
            "ising_constant": ising_constant,
            "candidate_batches": candidate_batches,
            "candidates_evaluated": candidate_batches * agents,
            "best_candidate_step": best_step,
            "final_mean_abs_transverse": mean_abs_y,
            "final_max_abs_transverse": max_abs_y,
            "optimality_proven": False,
        },
        "snapshots": snapshots,
    }
