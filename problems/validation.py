"""Validation helpers for generated problem instances.

Validation is deliberately independent of any solver.  It checks that a saved instance
is structurally sound, that its QUBO agrees with the problem script's own objective
calculation, and records characteristics that help build balanced benchmark datasets.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable

import networkx as nx
import numpy as np

from .common import qubo_energy


VALIDATOR_VERSION = 1
WARNING_CODES = {
    "all_items_fit",
    "all_sets_overlap",
    "complete_clique_graph",
    "complete_conflict_graph",
    "disconnected_graph",
    "dominant_partition_number",
    "duplicate_candidate_sets",
    "duplicate_clauses",
    "edgeless_clique_graph",
    "empty_graph",
    "excess_colors",
    "inactive_random_variables",
    "interaction_free_spin_glass",
    "isolated_vertices",
    "identical_partition_numbers",
    "literal_sign_imbalance",
    "no_item_fits",
    "no_quadratic_interactions",
    "no_set_overlaps",
    "not_k_colorable",
    "one_sided_coefficients",
    "unused_boolean_variables",
    "weak_ancilla_penalty",
    "weak_penalty",
    "zero_edge_penalty",
    "zero_one_hot_penalty",
}


def validation_result() -> dict:
    return {"errors": [], "warnings": [], "characteristics": {}}


def add_error(result: dict, code: str, message: str) -> None:
    result["errors"].append({"code": code, "message": message})


def add_warning(result: dict, code: str, message: str) -> None:
    result["warnings"].append({"code": code, "message": message})


def merge_validation(target: dict, addition: dict) -> None:
    target["errors"].extend(addition.get("errors", []))
    target["warnings"].extend(addition.get("warnings", []))
    target["characteristics"].update(addition.get("characteristics", {}))


def validate_problem(problem: dict, family_validator: Callable[[dict], dict]) -> dict:
    """Run generic and family-specific checks and return saved validation metadata."""
    result = _validate_qubo(problem.get("qubo"))
    try:
        family_result = family_validator(problem)
    except Exception as error:  # A validator failure must invalidate, not hide, an instance.
        family_result = validation_result()
        add_error(
            family_result,
            "validator_failure",
            f"The {problem.get('problem_type', 'problem')} validator failed: {error}",
        )
    merge_validation(result, family_result)
    result["status"] = "passed" if not result["errors"] else "failed"
    result["has_warnings"] = bool(result["warnings"])
    result["validator_version"] = VALIDATOR_VERSION
    return result


def check_encoding(
    problem: dict,
    expected_energy: Callable[[list[int]], float],
    *,
    exhaustive_limit: int = 12,
    random_samples: int = 128,
) -> dict:
    """Compare stored QUBO energy with an independent family objective calculation."""
    result = validation_result()
    qubo = problem["qubo"]
    num_variables = qubo["num_variables"]
    if num_variables <= exhaustive_limit:
        samples: Iterable[tuple[int, ...] | list[int]] = itertools.product(
            (0, 1), repeat=num_variables
        )
        method = "exhaustive"
        intended_count = 2**num_variables
    else:
        rng_seed = (int(problem.get("seed", 0)) + 0x51A7E) % (2**64)
        rng = np.random.default_rng(rng_seed)
        samples = [
            [0] * num_variables,
            [1] * num_variables,
            [index % 2 for index in range(num_variables)],
        ]
        samples.extend(
            [int(value) for value in row]
            for row in rng.integers(0, 2, size=(random_samples, num_variables))
        )
        method = "deterministic_random"
        intended_count = random_samples + 3

    checked = 0
    maximum_error = 0.0
    first_mismatch = None
    for raw_sample in samples:
        sample = list(raw_sample)
        expected = float(expected_energy(sample))
        actual = float(qubo_energy(qubo, sample))
        error = abs(expected - actual)
        maximum_error = max(maximum_error, error)
        tolerance = 1e-8 * max(1.0, abs(expected), abs(actual))
        if error > tolerance and first_mismatch is None:
            first_mismatch = {
                "sample": sample,
                "expected": expected,
                "actual": actual,
                "absolute_error": error,
            }
        checked += 1

    result["characteristics"]["encoding_check"] = {
        "method": method,
        "samples_checked": checked,
        "intended_samples": intended_count,
        "maximum_absolute_error": maximum_error,
    }
    if first_mismatch is not None:
        add_error(
            result,
            "encoding_mismatch",
            "Stored QUBO energy does not match the independently calculated problem energy "
            f"for sample {first_mismatch['sample']}: expected "
            f"{first_mismatch['expected']}, found {first_mismatch['actual']}.",
        )
    return result


def check_graph(num_vertices: int, edges: list, parameters: dict) -> tuple[dict, nx.Graph]:
    """Validate a simple undirected graph and record realized topology statistics."""
    result = validation_result()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    seen_pairs = set()
    for position, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) < 2:
            add_error(result, "malformed_edge", f"Edge {position} is not a valid edge record.")
            continue
        first, second = edge[0], edge[1]
        if not isinstance(first, int) or not isinstance(second, int):
            add_error(result, "noninteger_vertex", f"Edge {position} has a noninteger endpoint.")
            continue
        if not (0 <= first < num_vertices and 0 <= second < num_vertices):
            add_error(result, "edge_out_of_range", f"Edge {position} has an invalid endpoint.")
            continue
        if first == second:
            add_error(result, "self_loop", f"Edge {position} is a self-loop on vertex {first}.")
            continue
        pair = tuple(sorted((first, second)))
        if pair in seen_pairs:
            add_error(result, "duplicate_edge", f"Edge {pair} occurs more than once.")
            continue
        seen_pairs.add(pair)
        graph.add_edge(*pair)
        if len(edge) >= 3 and not _finite_number(edge[2]):
            add_error(result, "invalid_edge_weight", f"Edge {pair} has a non-finite weight.")

    possible_edges = num_vertices * (num_vertices - 1) // 2
    degrees = [degree for _, degree in graph.degree()]
    components = nx.number_connected_components(graph) if num_vertices else 0
    isolated = sum(degree == 0 for degree in degrees)
    result["characteristics"].update(
        {
            "graph_edges": graph.number_of_edges(),
            "graph_density": graph.number_of_edges() / possible_edges if possible_edges else 0.0,
            "graph_components": components,
            "isolated_vertices": isolated,
            "degree_min": min(degrees, default=0),
            "degree_max": max(degrees, default=0),
            "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
            "degree_std": float(np.std(degrees)) if degrees else 0.0,
        }
    )

    graph_type = parameters.get("graph_type")
    if graph_type == "complete" and graph.number_of_edges() != possible_edges:
        add_error(result, "wrong_complete_topology", "Generated complete graph is incomplete.")
    if graph_type == "cycle":
        expected_edges = 1 if num_vertices == 2 else num_vertices
        if graph.number_of_edges() != expected_edges or components != 1:
            add_error(result, "wrong_cycle_topology", "Generated cycle graph is malformed.")
    if graph_type == "random_regular":
        expected_degree = parameters.get("degree")
        if any(degree != expected_degree for degree in degrees):
            add_error(
                result,
                "wrong_regular_topology",
                "Generated random-regular graph does not have the requested degree.",
            )

    if graph.number_of_edges() == 0:
        add_warning(result, "empty_graph", "The graph has no edges.")
    if components > 1:
        add_warning(result, "disconnected_graph", f"The graph has {components} components.")
    if isolated:
        add_warning(result, "isolated_vertices", f"The graph has {isolated} isolated vertices.")
    return result, graph


def finite_statistics(values: Iterable[float]) -> dict:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def is_k_colorable(graph: nx.Graph, num_colors: int) -> bool:
    """Exact backtracking check intended only for small validation instances."""
    if num_colors < 1:
        return graph.number_of_nodes() == 0
    order = sorted(graph.nodes(), key=graph.degree, reverse=True)
    colors = {vertex: -1 for vertex in order}

    def assign(position: int) -> bool:
        if position == len(order):
            return True
        vertex = order[position]
        forbidden = {colors[neighbor] for neighbor in graph.neighbors(vertex)}
        for color in range(num_colors):
            if color not in forbidden:
                colors[vertex] = color
                if assign(position + 1):
                    return True
                colors[vertex] = -1
        return False

    return assign(0)


def _validate_qubo(qubo: dict | None) -> dict:
    result = validation_result()
    if not isinstance(qubo, dict):
        add_error(result, "missing_qubo", "Problem does not contain a QUBO mapping.")
        return result

    required = {"num_variables", "variable_names", "linear", "quadratic", "offset"}
    missing = sorted(required.difference(qubo))
    if missing:
        add_error(result, "incomplete_qubo", f"QUBO is missing: {', '.join(missing)}.")
        return result

    num_variables = qubo["num_variables"]
    if not isinstance(num_variables, int) or isinstance(num_variables, bool) or num_variables < 1:
        add_error(result, "invalid_variable_count", "QUBO variable count must be positive.")
        return result
    names = qubo["variable_names"]
    if not isinstance(names, list) or len(names) != num_variables:
        add_error(result, "invalid_variable_names", "Variable-name count does not match QUBO size.")
    elif len(set(names)) != len(names):
        add_error(result, "duplicate_variable_names", "QUBO variable names are not unique.")

    linear_indices = set()
    coefficients = []
    for position, term in enumerate(qubo["linear"]):
        if not isinstance(term, list) or len(term) != 2:
            add_error(result, "malformed_linear_term", f"Linear term {position} is malformed.")
            continue
        variable, coefficient = term
        if not isinstance(variable, int) or not 0 <= variable < num_variables:
            add_error(result, "linear_index_out_of_range", f"Linear term {position} has an invalid index.")
        if variable in linear_indices:
            add_error(result, "duplicate_linear_term", f"Variable {variable} has duplicate linear terms.")
        linear_indices.add(variable)
        if not _finite_number(coefficient):
            add_error(result, "nonfinite_coefficient", f"Linear term {position} is not finite.")
        else:
            coefficients.append(float(coefficient))

    quadratic_pairs = set()
    active_variables = set(linear_indices)
    for position, term in enumerate(qubo["quadratic"]):
        if not isinstance(term, list) or len(term) != 3:
            add_error(result, "malformed_quadratic_term", f"Quadratic term {position} is malformed.")
            continue
        first, second, coefficient = term
        if (
            not isinstance(first, int)
            or not isinstance(second, int)
            or not (0 <= first < second < num_variables)
        ):
            add_error(
                result,
                "invalid_quadratic_indices",
                f"Quadratic term {position} must use indices 0 <= first < second < n.",
            )
        pair = (first, second)
        if pair in quadratic_pairs:
            add_error(result, "duplicate_quadratic_term", f"Pair {pair} occurs more than once.")
        quadratic_pairs.add(pair)
        active_variables.update(pair)
        if not _finite_number(coefficient):
            add_error(result, "nonfinite_coefficient", f"Quadratic term {position} is not finite.")
        else:
            coefficients.append(float(coefficient))

    if not _finite_number(qubo["offset"]):
        add_error(result, "nonfinite_offset", "QUBO offset is not finite.")
    possible_pairs = num_variables * (num_variables - 1) // 2
    result["characteristics"].update(
        {
            "qubo_variables": num_variables,
            "qubo_linear_terms": len(qubo["linear"]),
            "qubo_quadratic_terms": len(qubo["quadratic"]),
            "qubo_quadratic_density": len(qubo["quadratic"]) / possible_pairs
            if possible_pairs
            else 0.0,
            "qubo_inactive_variables": num_variables - len(active_variables),
            "qubo_coefficient_statistics": finite_statistics(coefficients),
        }
    )
    return result


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
