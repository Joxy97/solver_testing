"""Maximum Independent Set QUBO generator."""

import numpy as np

from .common import (
    QuboBuilder,
    generate_graph,
    graph_parameter_schema,
    random_positive_integers,
)


NAME = "Maximum Independent Set"
DESCRIPTION = "Select a maximum-value set of vertices containing no adjacent pair."
PARAMETERS = {
    **graph_parameter_schema(),
    "weighted_vertices": {
        "type": "boolean",
        "default": False,
        "description": "Assign random values to vertices instead of value 1",
    },
    "min_vertex_value": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Smallest random vertex value",
    },
    "max_vertex_value": {
        "type": "integer",
        "default": 10,
        "min": 1,
        "description": "Largest random vertex value",
    },
    "penalty": {
        "type": "number",
        "default": None,
        "allow_none": True,
        "min": 0.0,
        "description": "Penalty per conflicting edge; blank chooses a safe value",
    },
}


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    edges = generate_graph(parameters, rng)
    num_vertices = parameters["num_vertices"]
    if parameters["min_vertex_value"] > parameters["max_vertex_value"]:
        raise ValueError("min_vertex_value cannot exceed max_vertex_value")
    values = (
        random_positive_integers(
            rng,
            num_vertices,
            parameters["min_vertex_value"],
            parameters["max_vertex_value"],
        )
        if parameters["weighted_vertices"]
        else [1] * num_vertices
    )
    penalty = parameters["penalty"]
    if penalty is None:
        penalty = float(max(values) + 1)

    builder = QuboBuilder(num_vertices, [f"vertex_{index}" for index in range(num_vertices)])
    for vertex, value in enumerate(values):
        builder.add_linear(vertex, -value)
    for first, second, _ in edges:
        builder.add_quadratic(first, second, penalty)

    resolved = dict(parameters)
    resolved["penalty"] = penalty
    return {
        "parameters": resolved,
        "qubo": builder.build(),
        "problem_data": {"edges": edges, "vertex_values": values},
        "encoding": {
            "description": "A value of 1 selects the vertex.",
            "energy_relationship": (
                "Energy is negative selected value plus penalty for each selected edge."
            ),
        },
    }


def validate(problem: dict) -> dict:
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
    num_vertices = parameters["num_vertices"]
    result, graph = check_graph(num_vertices, data["edges"], parameters)
    values = data["vertex_values"]
    if len(values) != num_vertices or any(value <= 0 for value in values):
        add_error(result, "invalid_vertex_values", "Every vertex must have one positive value.")
    if parameters["weighted_vertices"] and any(
        not parameters["min_vertex_value"] <= value <= parameters["max_vertex_value"]
        for value in values
    ):
        add_error(result, "vertex_value_out_of_range", "A vertex value is outside the requested range.")
    if graph.number_of_edges() == num_vertices * (num_vertices - 1) // 2:
        add_warning(
            result,
            "complete_conflict_graph",
            "Every vertex pair conflicts, so an independent set can contain at most one vertex.",
        )
    safe_threshold = max(values, default=0)
    if parameters["penalty"] <= safe_threshold:
        add_warning(
            result,
            "weak_penalty",
            "The edge penalty is not larger than every vertex value; an infeasible state may minimize the QUBO.",
        )
    result["characteristics"].update(
        {
            "vertex_value_statistics": finite_statistics(values),
            "conflicting_vertex_pairs": graph.number_of_edges(),
            "penalty_safe_threshold": float(safe_threshold),
            "penalty_margin": float(parameters["penalty"] - safe_threshold),
        }
    )

    def expected_energy(sample: list[int]) -> float:
        selected_value = sum(value * sample[index] for index, value in enumerate(values))
        conflicts = sum(sample[first] * sample[second] for first, second, _ in data["edges"])
        return -selected_value + parameters["penalty"] * conflicts

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
