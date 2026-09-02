"""Set Packing QUBO generator."""

import itertools

import numpy as np

from .common import QuboBuilder, random_positive_integers


NAME = "Set Packing"
DESCRIPTION = "Choose valuable sets that do not share any element."
PARAMETERS = {
    "universe_size": {
        "type": "integer",
        "default": 20,
        "min": 1,
        "description": "Number of elements in the universe",
    },
    "num_sets": {
        "type": "integer",
        "default": 30,
        "min": 1,
        "description": "Number of candidate sets",
    },
    "inclusion_probability": {
        "type": "number",
        "default": 0.15,
        "min": 0.0,
        "max": 1.0,
        "description": "Probability that an element belongs to a generated set",
    },
    "weighted_sets": {
        "type": "boolean",
        "default": True,
        "description": "Assign random values to sets instead of value 1",
    },
    "min_set_value": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Smallest generated set value",
    },
    "max_set_value": {
        "type": "integer",
        "default": 10,
        "min": 1,
        "description": "Largest generated set value",
    },
    "penalty": {
        "type": "number",
        "default": None,
        "allow_none": True,
        "min": 0.0,
        "description": "Overlap penalty; blank chooses a safe value",
    },
}


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    universe_size = parameters["universe_size"]
    num_sets = parameters["num_sets"]
    sets = []
    for _ in range(num_sets):
        members = [
            element
            for element in range(universe_size)
            if rng.random() < parameters["inclusion_probability"]
        ]
        if not members:
            members = [int(rng.integers(0, universe_size))]
        sets.append(members)

    if parameters["min_set_value"] > parameters["max_set_value"]:
        raise ValueError("min_set_value cannot exceed max_set_value")
    values = (
        random_positive_integers(
            rng,
            num_sets,
            parameters["min_set_value"],
            parameters["max_set_value"],
        )
        if parameters["weighted_sets"]
        else [1] * num_sets
    )
    penalty = parameters["penalty"]
    if penalty is None:
        penalty = float(max(values) + 1)

    overlaps = []
    member_sets = [set(members) for members in sets]
    for first, second in itertools.combinations(range(num_sets), 2):
        shared = sorted(member_sets[first].intersection(member_sets[second]))
        if shared:
            overlaps.append([first, second, shared])

    builder = QuboBuilder(num_sets, [f"set_{index}" for index in range(num_sets)])
    for set_index, value in enumerate(values):
        builder.add_linear(set_index, -value)
    for first, second, _ in overlaps:
        builder.add_quadratic(first, second, penalty)

    resolved = dict(parameters)
    resolved["penalty"] = penalty
    return {
        "parameters": resolved,
        "qubo": builder.build(),
        "problem_data": {
            "universe_size": universe_size,
            "sets": sets,
            "set_values": values,
            "overlaps": overlaps,
        },
        "encoding": {
            "description": "A value of 1 selects a set.",
            "energy_relationship": (
                "Energy is negative selected value plus one penalty per overlapping pair."
            ),
        },
    }


def validate(problem: dict) -> dict:
    from .validation import (
        add_error,
        add_warning,
        check_encoding,
        finite_statistics,
        merge_validation,
        validation_result,
    )

    result = validation_result()
    parameters = problem["parameters"]
    data = problem["problem_data"]
    universe_size = data["universe_size"]
    sets = data["sets"]
    values = data["set_values"]
    if len(sets) != parameters["num_sets"] or len(values) != len(sets):
        add_error(result, "invalid_set_count", "Set and value counts do not match num_sets.")
    for index, members in enumerate(sets):
        if not members or len(set(members)) != len(members):
            add_error(result, "invalid_candidate_set", f"Candidate set {index} is empty or contains duplicates.")
        if any(not isinstance(element, int) or not 0 <= element < universe_size for element in members):
            add_error(result, "set_element_out_of_range", f"Candidate set {index} contains an invalid element.")
    if any(value <= 0 for value in values):
        add_error(result, "invalid_set_value", "Every candidate set must have a positive value.")

    canonical_sets = [tuple(sorted(members)) for members in sets]
    duplicate_sets = len(canonical_sets) - len(set(canonical_sets))
    member_sets = [set(members) for members in sets]
    expected_overlaps = {
        (first, second): tuple(sorted(member_sets[first].intersection(member_sets[second])))
        for first, second in itertools.combinations(range(len(sets)), 2)
        if member_sets[first].intersection(member_sets[second])
    }
    saved_overlaps = {
        (first, second): tuple(shared) for first, second, shared in data["overlaps"]
    }
    if expected_overlaps != saved_overlaps:
        add_error(result, "invalid_overlap_data", "Stored set overlaps do not match candidate sets.")
    if duplicate_sets:
        add_warning(result, "duplicate_candidate_sets", f"There are {duplicate_sets} duplicate candidate sets.")
    if not expected_overlaps:
        add_warning(result, "no_set_overlaps", "No candidate sets overlap, so every set can be selected.")
    possible_pairs = len(sets) * (len(sets) - 1) // 2
    if possible_pairs and len(expected_overlaps) == possible_pairs:
        add_warning(result, "all_sets_overlap", "Every pair of candidate sets overlaps.")
    used_elements = set().union(*member_sets) if member_sets else set()
    safe_threshold = max(values, default=0)
    if parameters["penalty"] <= safe_threshold:
        add_warning(
            result,
            "weak_penalty",
            "Overlap penalty is not larger than every candidate-set value.",
        )
    result["characteristics"].update(
        {
            "set_size_statistics": finite_statistics(len(members) for members in sets),
            "set_value_statistics": finite_statistics(values),
            "duplicate_sets": duplicate_sets,
            "overlapping_set_pairs": len(expected_overlaps),
            "overlap_density": len(expected_overlaps) / possible_pairs if possible_pairs else 0.0,
            "unused_universe_elements": universe_size - len(used_elements),
            "penalty_safe_threshold": float(safe_threshold),
            "penalty_margin": float(parameters["penalty"] - safe_threshold),
        }
    )

    def expected_energy(sample: list[int]) -> float:
        selected_value = sum(value * sample[index] for index, value in enumerate(values))
        conflicts = sum(sample[first] * sample[second] for first, second in expected_overlaps)
        return -selected_value + parameters["penalty"] * conflicts

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
