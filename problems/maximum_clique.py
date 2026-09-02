"""Maximum Clique QUBO generator."""

import itertools

import numpy as np

from .common import (
    QuboBuilder,
    generate_graph,
    graph_parameter_schema,
    random_positive_integers,
)


NAME = "Maximum Clique"
DESCRIPTION = "Select a maximum-value collection of mutually adjacent vertices."
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
        "description": "Penalty per selected non-edge; blank chooses a safe value",
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

    edge_pairs = {(first, second) for first, second, _ in edges}
    non_edges = [
        [first, second]
        for first, second in itertools.combinations(range(num_vertices), 2)
        if (first, second) not in edge_pairs
    ]
    builder = QuboBuilder(num_vertices, [f"vertex_{index}" for index in range(num_vertices)])
    for vertex, value in enumerate(values):
        builder.add_linear(vertex, -value)
    for first, second in non_edges:
        builder.add_quadratic(first, second, penalty)

    resolved = dict(parameters)
    resolved["penalty"] = penalty
    return {
        "parameters": resolved,
        "qubo": builder.build(),
        "problem_data": {
            "edges": edges,
            "non_edges": non_edges,
            "vertex_values": values,
        },
        "encoding": {
            "description": "A value of 1 selects the vertex.",
            "energy_relationship": (
                "Energy is negative selected value plus penalty for each selected non-edge."
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
        add_error(result, "invalid_vertex_values", "Every clique vertex must have one positive value.")
    possible_pairs = num_vertices * (num_vertices - 1) // 2
    expected_non_edges = possible_pairs - graph.number_of_edges()
    if len(data["non_edges"]) != expected_non_edges:
        add_error(result, "invalid_non_edges", "Stored clique non-edges do not complement the graph.")
    if graph.number_of_edges() == 0:
        add_warning(
            result,
            "edgeless_clique_graph",
            "The graph contains no compatible pair, so a clique can contain at most one vertex.",
        )
    if graph.number_of_edges() == possible_pairs:
        add_warning(
            result,
            "complete_clique_graph",
            "Every vertex can be selected, making the maximum clique trivial.",
        )
    safe_threshold = max(values, default=0)
    if parameters["penalty"] <= safe_threshold:
        add_warning(
            result,
            "weak_penalty",
            "The non-edge penalty is not larger than every vertex value; an infeasible state may minimize the QUBO.",
        )
    result["characteristics"].update(
        {
            "vertex_value_statistics": finite_statistics(values),
            "clique_non_edges": len(data["non_edges"]),
            "clique_qubo_constraint_density": len(data["non_edges"]) / possible_pairs
            if possible_pairs
            else 0.0,
            "penalty_safe_threshold": float(safe_threshold),
            "penalty_margin": float(parameters["penalty"] - safe_threshold),
        }
    )

    def expected_energy(sample: list[int]) -> float:
        selected_value = sum(value * sample[index] for index, value in enumerate(values))
        violations = sum(sample[first] * sample[second] for first, second in data["non_edges"])
        return -selected_value + parameters["penalty"] * violations

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
