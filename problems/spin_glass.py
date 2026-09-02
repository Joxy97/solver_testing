"""Ising spin-glass instance and equivalent QUBO generator."""

import numpy as np

from .common import QuboBuilder, generate_graph, graph_parameter_schema


_GRAPH_SCHEMA = graph_parameter_schema()
NAME = "Spin Glass"
DESCRIPTION = "Generate an Ising model on a graph and store its equivalent binary QUBO."
PARAMETERS = {
    key: _GRAPH_SCHEMA[key]
    for key in ["num_vertices", "graph_type", "edge_probability", "degree"]
}
PARAMETERS.update(
    {
        "coupling_distribution": {
            "type": "choice",
            "default": "bimodal",
            "choices": ["bimodal", "gaussian"],
            "description": "Distribution for pair couplings J",
        },
        "coupling_scale": {
            "type": "number",
            "default": 1.0,
            "min": 0.000001,
            "description": "Magnitude or standard deviation of pair couplings",
        },
        "field_probability": {
            "type": "number",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "description": "Probability that a spin receives a local field",
        },
        "field_scale": {
            "type": "number",
            "default": 1.0,
            "min": 0.000001,
            "description": "Magnitude of nonzero local fields",
        },
    }
)


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    graph_parameters = {
        **parameters,
        "weighted": False,
        "min_edge_weight": 1,
        "max_edge_weight": 1,
    }
    topology = generate_graph(graph_parameters, rng)
    num_spins = parameters["num_vertices"]
    couplings = []
    for first, second, _ in topology:
        if parameters["coupling_distribution"] == "bimodal":
            coupling = float(rng.choice([-1.0, 1.0]) * parameters["coupling_scale"])
        else:
            coupling = float(rng.normal(0.0, parameters["coupling_scale"]))
        couplings.append([first, second, coupling])
    fields = []
    for spin in range(num_spins):
        if rng.random() < parameters["field_probability"]:
            field = float(rng.choice([-1.0, 1.0]) * parameters["field_scale"])
        else:
            field = 0.0
        fields.append(field)

    builder = QuboBuilder(num_spins, [f"spin_{index}" for index in range(num_spins)])
    # Substitute s_i = 2*x_i - 1 into sum h_i*s_i + sum J_ij*s_i*s_j.
    for spin, field in enumerate(fields):
        builder.add_linear(spin, 2.0 * field)
        builder.add_offset(-field)
    for first, second, coupling in couplings:
        builder.add_quadratic(first, second, 4.0 * coupling)
        builder.add_linear(first, -2.0 * coupling)
        builder.add_linear(second, -2.0 * coupling)
        builder.add_offset(coupling)

    return {
        "qubo": builder.build(),
        "problem_data": {
            "ising_convention": "H(s) = sum_i h_i*s_i + sum_{i<j} J_ij*s_i*s_j",
            "spin_mapping": "s_i = 2*x_i - 1",
            "fields": fields,
            "couplings": couplings,
        },
        "encoding": {
            "description": "Binary values map to spins by s = 2*x - 1.",
            "energy_relationship": "QUBO energy exactly equals the native Ising energy.",
        },
    }


def validate(problem: dict) -> dict:
    import networkx as nx

    from .validation import (
        add_error,
        add_warning,
        check_encoding,
        check_graph,
        finite_statistics,
        merge_validation,
    )

    parameters = problem["parameters"]
    data = problem["problem_data"]
    num_spins = parameters["num_vertices"]
    topology = [[first, second, coupling] for first, second, coupling in data["couplings"]]
    result, graph = check_graph(num_spins, topology, parameters)
    couplings = data["couplings"]
    fields = data["fields"]
    if len(fields) != num_spins or any(not np.isfinite(field) for field in fields):
        add_error(result, "invalid_spin_fields", "Spin Glass must contain one finite field per spin.")
    if any(not np.isfinite(coupling) for _, _, coupling in couplings):
        add_error(result, "invalid_couplings", "Every Spin Glass coupling must be finite.")
    if not couplings:
        add_warning(result, "interaction_free_spin_glass", "The Spin Glass has no pair interactions.")

    coupling_by_pair = {
        tuple(sorted((first, second))): coupling for first, second, coupling in couplings
    }
    cycles = nx.cycle_basis(graph)
    frustrated_cycles = 0
    for cycle in cycles:
        desired_product = 1.0
        for index, first in enumerate(cycle):
            second = cycle[(index + 1) % len(cycle)]
            coupling = coupling_by_pair[tuple(sorted((first, second)))]
            desired_product *= -1.0 if coupling > 0 else 1.0
        frustrated_cycles += desired_product < 0

    nonzero_fields = [field for field in fields if field != 0]
    positive_couplings = sum(coupling > 0 for _, _, coupling in couplings)
    result["characteristics"].update(
        {
            "coupling_statistics": finite_statistics(coupling for _, _, coupling in couplings),
            "field_statistics": finite_statistics(fields),
            "nonzero_fields": len(nonzero_fields),
            "realized_field_fraction": len(nonzero_fields) / num_spins,
            "positive_coupling_fraction": positive_couplings / len(couplings) if couplings else None,
            "cycle_basis_size": len(cycles),
            "frustrated_cycle_basis_count": frustrated_cycles,
            "frustrated_cycle_basis_fraction": frustrated_cycles / len(cycles) if cycles else None,
        }
    )

    def expected_energy(sample: list[int]) -> float:
        spins = [2 * value - 1 for value in sample]
        energy = sum(field * spins[index] for index, field in enumerate(fields))
        energy += sum(
            coupling * spins[first] * spins[second]
            for first, second, coupling in couplings
        )
        return energy

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
