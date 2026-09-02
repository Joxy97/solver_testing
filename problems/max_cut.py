"""Maximum Cut QUBO generator."""

import numpy as np

from .common import QuboBuilder, generate_graph, graph_parameter_schema


NAME = "Maximum Cut"
DESCRIPTION = "Partition graph vertices to maximize the total weight of crossing edges."
PARAMETERS = graph_parameter_schema()


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    edges = generate_graph(parameters, rng)
    num_vertices = parameters["num_vertices"]
    builder = QuboBuilder(num_vertices, [f"vertex_{index}" for index in range(num_vertices)])

    # Minimize the negative cut value: -w * (x_u + x_v - 2*x_u*x_v).
    for first, second, weight in edges:
        builder.add_linear(first, -weight)
        builder.add_linear(second, -weight)
        builder.add_quadratic(first, second, 2.0 * weight)

    return {
        "qubo": builder.build(),
        "problem_data": {"edges": edges},
        "encoding": {
            "description": "Binary value indicates which side of the cut contains a vertex.",
            "energy_relationship": "QUBO energy equals negative cut weight.",
        },
    }


def validate(problem: dict) -> dict:
    from .validation import (
        add_error,
        check_encoding,
        check_graph,
        finite_statistics,
        merge_validation,
    )

    result, _ = check_graph(
        problem["parameters"]["num_vertices"],
        problem["problem_data"]["edges"],
        problem["parameters"],
    )
    edges = problem["problem_data"]["edges"]
    weights = [weight for _, _, weight in edges]
    if any(weight <= 0 for weight in weights):
        add_error(result, "nonpositive_cut_weight", "All cut-edge weights must be positive.")
    if problem["parameters"]["weighted"] and any(
        not problem["parameters"]["min_edge_weight"]
        <= weight
        <= problem["parameters"]["max_edge_weight"]
        for weight in weights
    ):
        add_error(result, "cut_weight_out_of_range", "A cut-edge weight is outside the requested range.")
    result["characteristics"].update(
        {
            "total_edge_weight": float(sum(weights)),
            "edge_weight_statistics": finite_statistics(weights),
        }
    )

    def expected_energy(sample: list[int]) -> float:
        return -sum(
            weight for first, second, weight in edges if sample[first] != sample[second]
        )

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
