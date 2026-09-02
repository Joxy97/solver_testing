"""PyTorch simulated-bifurcation adapter with optional CUDA execution."""

from __future__ import annotations

import time

import numpy as np

from .common import (
    SolverCapabilityError,
    equally_spaced_targets,
    package_version,
    require_package,
    snapshot_record,
)


NAME = "Simulated Bifurcation"
DESCRIPTION = """
Uses a PyTorch implementation of ballistic or discrete simulated bifurcation. It is a
dense-matrix heuristic and can run on CPU or on a CUDA-capable GPU.
"""

PARAMETERS = {
    "agents": {
        "type": "integer",
        "default": 128,
        "min": 1,
        "description": "Independent oscillator systems evolved in parallel",
    },
    "max_steps": {
        "type": "integer",
        "default": 10_000,
        "min": 1,
        "description": "Maximum symplectic-Euler evolution steps",
    },
    "timeout": {
        "type": "number",
        "default": 60.0,
        "min": 0.0,
        "allow_none": True,
        "description": "Maximum optimizer time in seconds; null disables this stopping rule",
    },
    "mode": {
        "type": "choice",
        "default": "ballistic",
        "choices": ["ballistic", "discrete"],
        "description": "Ballistic is usually faster; discrete is usually more accurate",
    },
    "heated": {
        "type": "boolean",
        "default": False,
        "description": "Add thermal fluctuations intended to escape local optima",
    },
    "early_stopping": {
        "type": "boolean",
        "default": True,
        "description": "Freeze agents whose sampled energy remains stable",
    },
    "sampling_period": {
        "type": "integer",
        "default": 50,
        "min": 1,
        "description": "Evolution steps between convergence checks",
    },
    "convergence_threshold": {
        "type": "integer",
        "default": 50,
        "min": 1,
        "description": "Unchanged convergence checks required before an agent is frozen",
    },
    "dtype": {
        "type": "choice",
        "default": "float32",
        "choices": ["float32", "float64"],
        "description": "PyTorch coefficient and computation precision",
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
        "description": "PyTorch random seed used to initialize the agents",
    },
    "max_variables": {
        "type": "integer",
        "default": 5_000,
        "min": 1,
        "description": "Safety limit for the solver's dense n-by-n matrix allocation",
    },
    "num_snapshots": {
        "type": "integer",
        "default": 0,
        "min": 0,
        "description": "Equally spaced best-solution step checkpoints; zero disables snapshots",
    },
}


def version() -> str:
    return f"simulated-bifurcation {package_version('simulated-bifurcation')}"


def solve(qubo: dict, parameters: dict) -> dict:
    started = time.perf_counter()
    require_package("torch", "torch", NAME)
    require_package(
        "simulated_bifurcation", "simulated-bifurcation", NAME
    )
    import simulated_bifurcation as sb
    import torch

    num_variables = int(qubo["num_variables"])
    if num_variables > parameters["max_variables"]:
        element_bytes = 4 if parameters["dtype"] == "float32" else 8
        estimated_mib = num_variables * num_variables * element_bytes / 1024**2
        raise SolverCapabilityError(
            f"Simulated bifurcation needs a dense {num_variables:,} x {num_variables:,} "
            f"matrix (about {estimated_mib:,.1f} MiB before working memory), above "
            f"max_variables={parameters['max_variables']:,}."
        )

    requested_device = parameters["device"].strip().lower()
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cpu" or requested_device == "cuda" or requested_device.startswith(
        "cuda:"
    ):
        device = requested_device
    else:
        raise ValueError("device must be 'cpu', 'auto', 'cuda', or a CUDA index such as 'cuda:0'")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SolverCapabilityError(
            "CUDA was requested for simulated bifurcation, but this PyTorch installation "
            "cannot access a CUDA GPU. Use device=cpu or install a CUDA-enabled PyTorch build."
        )

    numpy_dtype = np.float32 if parameters["dtype"] == "float32" else np.float64
    torch_dtype = torch.float32 if parameters["dtype"] == "float32" else torch.float64
    matrix = np.zeros((num_variables, num_variables), dtype=numpy_dtype)
    linear = np.zeros(num_variables, dtype=numpy_dtype)
    for variable, coefficient in qubo["linear"]:
        linear[int(variable)] += float(coefficient)
    for first, second, coefficient in qubo["quadratic"]:
        # A symmetric half-coefficient representation gives x.T @ matrix @ x = q*x_i*x_j.
        half = float(coefficient) / 2.0
        matrix[int(first), int(second)] += half
        matrix[int(second), int(first)] += half

    matrix_tensor = torch.as_tensor(matrix, dtype=torch_dtype, device=device)
    linear_tensor = torch.as_tensor(linear, dtype=torch_dtype, device=device)

    num_snapshots = parameters["num_snapshots"]
    snapshot_targets = equally_spaced_targets(
        parameters["max_steps"], num_snapshots
    )
    run_targets = snapshot_targets or [parameters["max_steps"]]
    snapshots = []
    vector = evaluation = None
    sample = None
    for index, step_limit in enumerate(run_targets, start=1):
        torch.manual_seed(parameters["seed"])
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(parameters["seed"])
        checkpoint_timeout = parameters["timeout"]
        if num_snapshots and checkpoint_timeout is not None:
            checkpoint_timeout *= step_limit / parameters["max_steps"]
        vector, evaluation = sb.minimize(
            matrix_tensor,
            linear_tensor,
            float(qubo.get("offset", 0.0)),
            domain="binary",
            dtype=torch_dtype,
            device=device,
            agents=parameters["agents"],
            max_steps=step_limit,
            best_only=True,
            mode=parameters["mode"],
            heated=parameters["heated"],
            verbose=False,
            early_stopping=parameters["early_stopping"],
            sampling_period=parameters["sampling_period"],
            convergence_threshold=parameters["convergence_threshold"],
            timeout=checkpoint_timeout,
        )
        sample = (
            vector.detach()
            .to(device="cpu")
            .round()
            .to(dtype=torch.int8)
            .reshape(-1)
            .tolist()
        )
        if num_snapshots:
            snapshots.append(
                snapshot_record(
                    index=index,
                    count=num_snapshots,
                    unit="evolution_steps",
                    target=step_limit,
                    total=parameters["max_steps"],
                    sample=sample,
                    reported_energy=float(evaluation.detach().cpu().item()),
                    elapsed_seconds=time.perf_counter() - started,
                    metrics={
                        "step_limit": step_limit,
                        "timeout_limit": checkpoint_timeout,
                    },
                )
            )
    device_name = "cpu"
    if device.startswith("cuda"):
        device_name = f"{device} ({torch.cuda.get_device_name(torch.device(device))})"

    return {
        "status": "completed",
        "sample": sample,
        "reported_energy": float(evaluation.detach().cpu().item()),
        "device": device_name,
        "metrics": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "dense_matrix_mib": matrix.nbytes / 1024**2,
            "optimality_proven": False,
        },
        "snapshots": snapshots,
    }
