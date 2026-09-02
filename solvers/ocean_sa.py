"""D-Wave Ocean SDK simulated annealing adapter."""

from __future__ import annotations

from .common import SolverCapabilityError, package_version, require_package


NAME = "D-Wave Ocean Simulated Annealing"
DESCRIPTION = """
Runs the classical SimulatedAnnealingSampler distributed with D-Wave's Ocean tools.
It produces heuristic samples on the local CPU and does not contact a quantum computer.
"""

PARAMETERS = {
    "num_reads": {
        "type": "integer",
        "default": 100,
        "min": 1,
        "description": "Independent annealing runs returned by the sampler",
    },
    "num_sweeps": {
        "type": "integer",
        "default": 1_000,
        "min": 1,
        "description": "Complete variable-update sweeps in each annealing run",
    },
    "num_sweeps_per_beta": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Sweeps performed at every inverse-temperature value",
    },
    "beta_schedule_type": {
        "type": "choice",
        "default": "geometric",
        "choices": ["geometric", "linear"],
        "description": "Interpolation used between the start and end inverse temperatures",
    },
    "beta_min": {
        "type": "number",
        "default": None,
        "min": 0.0,
        "allow_none": True,
        "description": "Starting inverse temperature; null lets Ocean choose from the QUBO biases",
    },
    "beta_max": {
        "type": "number",
        "default": None,
        "min": 0.0,
        "allow_none": True,
        "description": "Ending inverse temperature; null lets Ocean choose from the QUBO biases",
    },
    "randomize_order": {
        "type": "boolean",
        "default": False,
        "description": "Randomize the variable-update order within each sweep",
    },
    "proposal_acceptance_criteria": {
        "type": "choice",
        "default": "Metropolis",
        "choices": ["Metropolis", "Gibbs"],
        "description": "Rule used to accept proposed spin flips",
    },
    "seed": {
        "type": "integer",
        "default": 0,
        "min": 0,
        "max": 2_147_483_647,
        "description": "Pseudo-random seed for reproducible CPU runs",
    },
    "max_variable_updates": {
        "type": "integer",
        "default": 100_000_000,
        "min": 1,
        "description": "Safety limit on num_variables times num_reads times num_sweeps",
    },
}


def version() -> str:
    return f"dwave-samplers {package_version('dwave-samplers')}"


def solve(qubo: dict, parameters: dict) -> dict:
    require_package("dimod", "dwave-ocean-sdk", NAME)
    require_package("dwave.samplers", "dwave-ocean-sdk", NAME)
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler

    beta_values = (parameters["beta_min"], parameters["beta_max"])
    if (beta_values[0] is None) != (beta_values[1] is None):
        raise ValueError("beta_min and beta_max must either both be set or both be null")
    if beta_values[0] is not None and beta_values[0] >= beta_values[1]:
        raise ValueError("beta_min must be smaller than beta_max")

    num_variables = int(qubo["num_variables"])
    planned_updates = (
        num_variables * parameters["num_reads"] * parameters["num_sweeps"]
    )
    if planned_updates > parameters["max_variable_updates"]:
        raise SolverCapabilityError(
            f"Ocean SA would attempt approximately {planned_updates:,} variable updates, "
            f"above max_variable_updates={parameters['max_variable_updates']:,}. Reduce "
            "num_reads or num_sweeps, or deliberately raise the safety limit."
        )
    linear = {variable: 0.0 for variable in range(num_variables)}
    for variable, coefficient in qubo["linear"]:
        linear[int(variable)] += float(coefficient)
    quadratic = {
        (int(first), int(second)): float(coefficient)
        for first, second, coefficient in qubo["quadratic"]
    }
    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        float(qubo.get("offset", 0.0)),
        dimod.BINARY,
    )

    sampler = SimulatedAnnealingSampler()
    sample_kwargs = {
        "num_reads": parameters["num_reads"],
        "num_sweeps": parameters["num_sweeps"],
        "num_sweeps_per_beta": parameters["num_sweeps_per_beta"],
        "beta_schedule_type": parameters["beta_schedule_type"],
        "seed": parameters["seed"],
        "randomize_order": parameters["randomize_order"],
        "proposal_acceptance_criteria": parameters["proposal_acceptance_criteria"],
    }
    if beta_values[0] is not None:
        sample_kwargs["beta_range"] = list(beta_values)
    sampleset = sampler.sample(bqm, **sample_kwargs)
    best = sampleset.first
    sample = [int(best.sample[variable]) for variable in range(num_variables)]

    return {
        "status": "completed",
        "sample": sample,
        "reported_energy": float(best.energy),
        "device": "cpu",
        "metrics": {
            "samples_returned": len(sampleset),
            "best_num_occurrences": int(best.num_occurrences),
            "planned_variable_updates": planned_updates,
            "sampler_info": sampleset.info,
            "optimality_proven": False,
        },
    }
