"""Number Partitioning QUBO generator."""

import numpy as np

from .common import QuboBuilder, random_positive_integers


NAME = "Number Partitioning"
DESCRIPTION = "Split positive integers into two groups with the smallest possible sum difference."
PARAMETERS = {
    "num_numbers": {
        "type": "integer",
        "default": 20,
        "min": 2,
        "description": "Number of integers to partition",
    },
    "min_value": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Smallest generated integer",
    },
    "max_value": {
        "type": "integer",
        "default": 100,
        "min": 1,
        "description": "Largest generated integer",
    },
}


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    numbers = random_positive_integers(
        rng,
        parameters["num_numbers"],
        parameters["min_value"],
        parameters["max_value"],
    )
    names = [f"number_{index}" for index in range(len(numbers))]
    builder = QuboBuilder(len(numbers), names)
    total = sum(numbers)
    builder.add_square(
        {index: 2.0 * value for index, value in enumerate(numbers)},
        constant=-float(total),
    )
    return {
        "qubo": builder.build(),
        "problem_data": {"numbers": numbers},
        "encoding": {
            "description": "A value of 1 puts an integer in the first partition.",
            "energy_relationship": "QUBO energy equals the squared partition imbalance.",
        },
    }


def validate(problem: dict) -> dict:
    import math

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
    numbers = problem["problem_data"]["numbers"]
    if len(numbers) != parameters["num_numbers"] or any(value <= 0 for value in numbers):
        add_error(result, "invalid_partition_numbers", "Partition data must contain the requested number of positive integers.")
    if any(not parameters["min_value"] <= value <= parameters["max_value"] for value in numbers):
        add_error(result, "partition_value_out_of_range", "A partition number is outside the requested range.")
    total = sum(numbers)
    greatest_common_divisor = math.gcd(*numbers) if numbers else 0
    duplicate_fraction = 1.0 - len(set(numbers)) / len(numbers) if numbers else 0.0
    dominant_fraction = max(numbers, default=0) / total if total else 0.0
    if len(set(numbers)) == 1:
        add_warning(result, "identical_partition_numbers", "All generated numbers are identical.")
    if dominant_fraction > 0.5:
        add_warning(result, "dominant_partition_number", "One number is larger than half of the total sum.")

    perfect_partition = None
    if len(numbers) <= 24 and total <= 1_000_000:
        reachable = {0}
        for value in numbers:
            reachable |= {subtotal + value for subtotal in tuple(reachable)}
        perfect_partition = total % 2 == 0 and total // 2 in reachable
    result["characteristics"].update(
        {
            "number_statistics": finite_statistics(numbers),
            "number_sum": total,
            "number_bit_length": max(numbers, default=0).bit_length(),
            "numbers_gcd": greatest_common_divisor,
            "duplicate_number_fraction": duplicate_fraction,
            "largest_number_fraction_of_sum": dominant_fraction,
            "perfect_partition_exists": perfect_partition,
        }
    )

    def expected_energy(sample: list[int]) -> float:
        first_sum = sum(value * sample[index] for index, value in enumerate(numbers))
        return float((2 * first_sum - total) ** 2)

    merge_validation(result, check_encoding(problem, expected_energy))
    return result
