"""Graph Coloring QUBO generator."""

import numpy as np

from .common import QuboBuilder, generate_graph, graph_parameter_schema


NAME = "Graph Coloring"
DESCRIPTION = "Assign one color to each vertex while keeping adjacent colors different."
PARAMETERS = {
    **graph_parameter_schema(),
    "num_colors": {
        "type": "integer",
        "default": 3,
        "min": 1,
        "description": "Number of available colors",
    },
    "one_color_penalty": {
        "type": "number",
        "default": 2.0,
        "min": 0.0,
        "description": "Penalty for assigning other than one color to a vertex",
    },
    "edge_penalty": {
        "type": "number",
        "default": 1.0,
        "min": 0.0,
        "description": "Penalty when adjacent vertices have the same color",
    },
}


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    edges = generate_graph(parameters, rng)
    num_vertices = parameters["num_vertices"]
    num_colors = parameters["num_colors"]
    names = [
        f"vertex_{vertex}_color_{color}"
        for vertex in range(num_vertices)
        for color in range(num_colors)
    ]
    builder = QuboBuilder(num_vertices * num_colors, names)

    def variable(vertex: int, color: int) -> int:
        return vertex * num_colors + color

    for vertex in range(num_vertices):
        coefficients = {variable(vertex, color): 1.0 for color in range(num_colors)}
        builder.add_square(coefficients, constant=-1.0, weight=parameters["one_color_penalty"])
    for first, second, _ in edges:
        for color in range(num_colors):
            builder.add_quadratic(
                variable(first, color),
                variable(second, color),
                parameters["edge_penalty"],
            )

    return {
        "qubo": builder.build(),
        "problem_data": {"edges": edges, "num_colors": num_colors},
        "encoding": {
            "description": "One-hot variables encode the color assigned to every vertex.",
            "energy_relationship": (
                "Energy is the sum of one-hot and equal-adjacent-color penalties."
            ),
        },
    }


def validate(problem: dict) -> dict:
    from .validation import (
        add_warning,
        check_encoding,
        check_graph,
        is_k_colorable,
        merge_validation,
    )

    parameters = problem["parameters"]
    data = problem["problem_data"]
    num_vertices = parameters["num_vertices"]
    num_colors = parameters["num_colors"]
    result, graph = check_graph(num_vertices, data["edges"], parameters)
    if num_colors >= num_vertices:
        add_warning(
            result,
            "excess_colors",
            "At least one distinct color is available per vertex, so a valid coloring is immediate.",
        )
    if parameters["one_color_penalty"] == 0:
        add_warning(result, "zero_one_hot_penalty", "The QUBO does not enforce one color per vertex.")
    if parameters["edge_penalty"] == 0 and graph.number_of_edges():
        add_warning(result, "zero_edge_penalty", "The QUBO does not enforce different colors on adjacent vertices.")
    colorable = None
    if num_vertices <= 16:
        colorable = is_k_colorable(graph, num_colors)
        if not colorable:
            add_warning(
                result,
                "not_k_colorable",
                "Exact small-instance validation found that the graph cannot be colored with the requested colors.",
            )
    result["characteristics"].update(
        {
            "colors": num_colors,
            "coloring_variables": num_vertices * num_colors,
            "colorable_when_checked": colorable,
            "one_color_to_edge_penalty_ratio": (
                parameters["one_color_penalty"] / parameters["edge_penalty"]
                if parameters["edge_penalty"]
                else None
            ),
        }
    )

    def expected_energy(sample: list[int]) -> float:
        energy = 0.0
        for vertex in range(num_vertices):
            assigned = sum(
                sample[vertex * num_colors + color] for color in range(num_colors)
            )
            energy += parameters["one_color_penalty"] * (assigned - 1) ** 2
        energy += parameters["edge_penalty"] * sum(
            sample[first * num_colors + color] * sample[second * num_colors + color]
            for first, second, _ in data["edges"]
            for color in range(num_colors)
        )
        return energy

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
