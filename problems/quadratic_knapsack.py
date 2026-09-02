"""Quadratic Knapsack QUBO generator."""

import itertools

import numpy as np

from .common import QuboBuilder, random_positive_integers, slack_weights


NAME = "Quadratic Knapsack"
DESCRIPTION = "Choose capacity-limited items with linear values and pairwise bonus values."
PARAMETERS = {
    "num_items": {
        "type": "integer",
        "default": 15,
        "min": 1,
        "description": "Number of candidate items",
    },
    "min_weight": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Smallest item weight",
    },
    "max_weight": {
        "type": "integer",
        "default": 20,
        "min": 1,
        "description": "Largest item weight",
    },
    "min_value": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Smallest linear item value",
    },
    "max_value": {
        "type": "integer",
        "default": 30,
        "min": 1,
        "description": "Largest linear item value",
    },
    "quadratic_density": {
        "type": "number",
        "default": 0.25,
        "min": 0.0,
        "max": 1.0,
        "description": "Fraction of item pairs receiving a bonus",
    },
    "min_quadratic_value": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Smallest pairwise bonus",
    },
    "max_quadratic_value": {
        "type": "integer",
        "default": 10,
        "min": 1,
        "description": "Largest pairwise bonus",
    },
    "capacity_ratio": {
        "type": "number",
        "default": 0.5,
        "min": 0.01,
        "max": 1.0,
        "description": "Capacity as a fraction of total item weight",
    },
    "penalty": {
        "type": "number",
        "default": None,
        "allow_none": True,
        "min": 0.0,
        "description": "Capacity penalty; blank chooses a conservative value",
    },
}


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    num_items = parameters["num_items"]
    weights = random_positive_integers(
        rng, num_items, parameters["min_weight"], parameters["max_weight"]
    )
    values = random_positive_integers(
        rng, num_items, parameters["min_value"], parameters["max_value"]
    )
    if parameters["min_quadratic_value"] > parameters["max_quadratic_value"]:
        raise ValueError("min_quadratic_value cannot exceed max_quadratic_value")

    quadratic_values = []
    for first, second in itertools.combinations(range(num_items), 2):
        if rng.random() < parameters["quadratic_density"]:
            value = int(
                rng.integers(
                    parameters["min_quadratic_value"],
                    parameters["max_quadratic_value"] + 1,
                )
            )
            quadratic_values.append([first, second, value])

    capacity = max(1, int(sum(weights) * parameters["capacity_ratio"]))
    slack = slack_weights(capacity)
    total_possible_value = sum(values) + sum(value for _, _, value in quadratic_values)
    penalty = parameters["penalty"]
    if penalty is None:
        penalty = float(total_possible_value + 1)

    variable_names = [f"item_{index}" for index in range(num_items)]
    variable_names += [f"capacity_slack_{index}" for index in range(len(slack))]
    builder = QuboBuilder(len(variable_names), variable_names)
    for item, value in enumerate(values):
        builder.add_linear(item, -value)
    for first, second, value in quadratic_values:
        builder.add_quadratic(first, second, -value)

    capacity_coefficients = {item: float(weight) for item, weight in enumerate(weights)}
    for offset, slack_weight in enumerate(slack):
        capacity_coefficients[num_items + offset] = float(slack_weight)
    builder.add_square(capacity_coefficients, constant=-float(capacity), weight=penalty)

    resolved = dict(parameters)
    resolved["penalty"] = penalty
    return {
        "parameters": resolved,
        "qubo": builder.build(),
        "problem_data": {
            "item_weights": weights,
            "item_values": values,
            "quadratic_values": quadratic_values,
            "capacity": capacity,
            "slack_weights": slack,
            "num_item_variables": num_items,
        },
        "encoding": {
            "description": (
                "Item variables select objects; extra binary variables encode unused capacity."
            ),
            "energy_relationship": (
                "Energy is negative total value plus penalty times squared capacity residual."
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
    num_items = parameters["num_items"]
    weights = data["item_weights"]
    values = data["item_values"]
    bonuses = data["quadratic_values"]
    capacity = data["capacity"]
    slack = data["slack_weights"]
    if len(weights) != num_items or any(weight <= 0 for weight in weights):
        add_error(result, "invalid_item_weights", "Knapsack must contain one positive weight per item.")
    if len(values) != num_items or any(value <= 0 for value in values):
        add_error(result, "invalid_item_values", "Knapsack must contain one positive value per item.")
    if not 1 <= capacity <= sum(weights):
        add_error(result, "invalid_capacity", "Knapsack capacity must lie between 1 and total item weight.")

    seen_pairs = set()
    for first, second, bonus in bonuses:
        pair = (first, second)
        if not 0 <= first < second < num_items or pair in seen_pairs or bonus <= 0:
            add_error(result, "invalid_quadratic_bonus", "Quadratic item bonuses contain an invalid pair or value.")
        seen_pairs.add(pair)
    reachable_slack = {0}
    for slack_weight in slack:
        reachable_slack |= {value + slack_weight for value in tuple(reachable_slack)}
    if any(value not in reachable_slack for value in range(capacity + 1)):
        add_error(result, "invalid_slack_encoding", "Slack variables cannot represent every unused capacity value.")

    items_that_fit = sum(weight <= capacity for weight in weights)
    if items_that_fit == 0:
        add_warning(result, "no_item_fits", "No generated item fits inside the knapsack capacity.")
    if sum(weights) <= capacity:
        add_warning(result, "all_items_fit", "All generated items fit simultaneously, so capacity is nonbinding.")
    total_possible_value = sum(values) + sum(bonus for _, _, bonus in bonuses)
    if parameters["penalty"] <= total_possible_value:
        add_warning(
            result,
            "weak_penalty",
            "Capacity penalty does not exceed the total possible positive objective value.",
        )
    possible_pairs = num_items * (num_items - 1) // 2
    result["characteristics"].update(
        {
            "item_weight_statistics": finite_statistics(weights),
            "item_value_statistics": finite_statistics(values),
            "quadratic_bonus_statistics": finite_statistics(bonus for _, _, bonus in bonuses),
            "quadratic_bonus_density": len(bonuses) / possible_pairs if possible_pairs else 0.0,
            "capacity": capacity,
            "capacity_fraction_of_total_weight": capacity / sum(weights) if weights else None,
            "items_fitting_individually": items_that_fit,
            "slack_variables": len(slack),
            "penalty_safe_threshold": float(total_possible_value),
            "penalty_margin": float(parameters["penalty"] - total_possible_value),
        }
    )

    def expected_energy(sample: list[int]) -> float:
        item_sample = sample[:num_items]
        slack_sample = sample[num_items:]
        objective = sum(value * item_sample[index] for index, value in enumerate(values))
        objective += sum(
            bonus * item_sample[first] * item_sample[second]
            for first, second, bonus in bonuses
        )
        residual = sum(weight * item_sample[index] for index, weight in enumerate(weights))
        residual += sum(weight * bit for weight, bit in zip(slack, slack_sample))
        residual -= capacity
        return -objective + parameters["penalty"] * residual**2

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
