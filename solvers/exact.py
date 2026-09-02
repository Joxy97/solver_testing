"""Streaming exhaustive enumeration for small QUBOs."""

from __future__ import annotations

import time

import numpy as np

from .common import SolverCapabilityError, equally_spaced_targets, snapshot_record


NAME = "Exact Enumeration"
DESCRIPTION = """
Enumerates every binary assignment in bounded-memory batches and returns a proven
global optimum. This is a ground-truth solver for small QUBOs, not a scalable solver.
"""

PARAMETERS = {
    "max_variables": {
        "type": "integer",
        "default": 24,
        "min": 1,
        "max": 62,
        "description": "Safety limit; reject larger QUBOs before enumerating 2^n states",
    },
    "batch_size": {
        "type": "integer",
        "default": 65_536,
        "min": 1,
        "description": "Assignments evaluated together; larger batches trade memory for speed",
    },
    "num_snapshots": {
        "type": "integer",
        "default": 0,
        "min": 0,
        "description": "Equally spaced best-so-far checkpoints; zero disables snapshots",
    },
}


def version() -> str:
    return "built in"


def solve(qubo: dict, parameters: dict) -> dict:
    started = time.perf_counter()
    num_variables = int(qubo["num_variables"])
    if num_variables > parameters["max_variables"]:
        raise SolverCapabilityError(
            f"Exact enumeration is limited to {parameters['max_variables']} variables by "
            f"max_variables, but this QUBO has {num_variables}. Increase the limit only if "
            "you intentionally want to enumerate all 2^n states."
        )
    if num_variables > 62:
        raise SolverCapabilityError(
            "This exact implementation uses 64-bit assignment indices and supports at most "
            "62 variables."
        )

    linear = np.zeros(num_variables, dtype=np.float64)
    for variable, coefficient in qubo["linear"]:
        linear[int(variable)] += float(coefficient)

    if qubo["quadratic"]:
        first = np.asarray([term[0] for term in qubo["quadratic"]], dtype=np.intp)
        second = np.asarray([term[1] for term in qubo["quadratic"]], dtype=np.intp)
        quadratic = np.asarray([term[2] for term in qubo["quadratic"]], dtype=np.float64)
    else:
        first = second = np.empty(0, dtype=np.intp)
        quadratic = np.empty(0, dtype=np.float64)

    total_states = 1 << num_variables
    snapshot_targets = equally_spaced_targets(
        total_states, parameters["num_snapshots"]
    )
    next_snapshot = 0
    snapshots = []
    batch_size = min(parameters["batch_size"], total_states)
    bit_positions = np.arange(num_variables, dtype=np.uint64)
    best_energy = float("inf")
    best_index = 0
    best_count = 0

    start = 0
    while start < total_states:
        stop = min(start + batch_size, total_states)
        if next_snapshot < len(snapshot_targets):
            stop = min(stop, snapshot_targets[next_snapshot])
        indices = np.arange(start, stop, dtype=np.uint64)
        samples = ((indices[:, None] >> bit_positions) & 1).astype(np.uint8)
        energies = samples @ linear

        # Chunking prevents a dense QUBO from creating a very large B x m temporary.
        for term_start in range(0, len(quadratic), 256):
            term_stop = min(term_start + 256, len(quadratic))
            products = (
                samples[:, first[term_start:term_stop]]
                * samples[:, second[term_start:term_stop]]
            )
            energies += products @ quadratic[term_start:term_stop]

        local_position = int(np.argmin(energies))
        local_energy = float(energies[local_position])
        if local_energy < best_energy:
            best_energy = local_energy
            best_index = int(indices[local_position])
            best_count = int(np.count_nonzero(energies == local_energy))
        elif local_energy == best_energy:
            best_count += int(np.count_nonzero(energies == best_energy))

        if next_snapshot < len(snapshot_targets) and stop == snapshot_targets[next_snapshot]:
            checkpoint_sample = [
                int((best_index >> variable) & 1) for variable in range(num_variables)
            ]
            snapshots.append(
                snapshot_record(
                    index=next_snapshot + 1,
                    count=len(snapshot_targets),
                    unit="states",
                    target=stop,
                    total=total_states,
                    sample=checkpoint_sample,
                    reported_energy=best_energy + float(qubo.get("offset", 0.0)),
                    elapsed_seconds=time.perf_counter() - started,
                    metrics={"states_evaluated": stop},
                )
            )
            next_snapshot += 1
        start = stop

    sample = [int((best_index >> variable) & 1) for variable in range(num_variables)]
    best_energy += float(qubo.get("offset", 0.0))
    return {
        "status": "optimal",
        "sample": sample,
        "reported_energy": best_energy,
        "device": "cpu",
        "metrics": {
            "states_evaluated": total_states,
            "optimal_assignments_found": best_count,
            "optimality_proven": True,
        },
        "snapshots": snapshots,
    }
