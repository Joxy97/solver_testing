"""Generic random QUBO generator."""

import itertools
import math

import numpy as np

from .common import QuboBuilder


NAME = "Random QUBO"
DESCRIPTION = "Generate a QUBO directly without an underlying combinatorial problem."
PARAMETERS = {
    "num_variables": {
        "type": "integer",
        "default": 50,
        "min": 1,
        "description": "Number of binary variables",
    },
    "quadratic_density": {
        "type": "number",
        "default": 0.25,
        "min": 0.0,
        "max": 1.0,
        "description": "Probability that a variable pair has a coefficient",
    },
    "distribution": {
        "type": "choice",
        "default": "integer",
        "choices": ["integer", "uniform", "normal"],
        "description": "Coefficient distribution",
    },
    "coefficient_min": {
        "type": "number",
        "default": -10.0,
        "description": "Minimum for integer and uniform distributions",
    },
    "coefficient_max": {
        "type": "number",
        "default": 10.0,
        "description": "Maximum for integer and uniform distributions",
    },
    "normal_scale": {
        "type": "number",
        "default": 5.0,
        "min": 0.000001,
        "description": "Standard deviation for the normal distribution",
    },
    "zero_linear_probability": {
        "type": "number",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "description": "Probability that a linear coefficient is zero",
    },
}


def _coefficient(parameters: dict, rng: np.random.Generator) -> float:
    distribution = parameters["distribution"]
    minimum = parameters["coefficient_min"]
    maximum = parameters["coefficient_max"]
    if minimum > maximum:
        raise ValueError("coefficient_min cannot exceed coefficient_max")
    if distribution == "integer":
        low = int(minimum)
        high = int(maximum)
        if low != minimum or high != maximum:
            raise ValueError("integer distribution requires integer coefficient bounds")
        return float(rng.integers(low, high + 1))
    if distribution == "uniform":
        return float(rng.uniform(minimum, maximum))
    return float(rng.normal(0.0, parameters["normal_scale"]))


def _pair_prefix(num_variables: int, first: int) -> int:
    """Return the number of upper-triangular pairs before row `first`."""
    return first * (2 * num_variables - first - 1) // 2


def _pair_from_index(num_variables: int, pair_index: int) -> tuple[int, int]:
    """Map a flattened upper-triangular index to an `(i, j)` pair."""
    discriminant = (2 * num_variables - 1) ** 2 - 8 * pair_index
    first = int(((2 * num_variables - 1) - math.sqrt(discriminant)) // 2)
    while _pair_prefix(num_variables, first + 1) <= pair_index:
        first += 1
    while _pair_prefix(num_variables, first) > pair_index:
        first -= 1
    second = first + 1 + pair_index - _pair_prefix(num_variables, first)
    return first, second


def _sample_quadratic_pairs(
    num_variables: int, density: float, rng: np.random.Generator
):
    """Sample independent pairs efficiently for both ordinary and large sparse QUBOs."""
    total_pairs = num_variables * (num_variables - 1) // 2
    if density <= 0.0:
        return
    if density >= 1.0:
        yield from itertools.combinations(range(num_variables), 2)
        return

    # Keep the original, direct method for ordinary sizes. Geometric gaps give the same
    # independent Bernoulli distribution for large sparse QUBOs without scanning O(n²)
    # absent interactions.
    if total_pairs <= 1_000_000:
        for first, second in itertools.combinations(range(num_variables), 2):
            if rng.random() < density:
                yield first, second
        return

    pair_index = -1
    while True:
        pair_index += int(rng.geometric(density))
        if pair_index >= total_pairs:
            return
        yield _pair_from_index(num_variables, pair_index)


def generate(parameters: dict, seed: int, progress=None) -> dict:
    rng = np.random.default_rng(seed)
    num_variables = parameters["num_variables"]
    builder = QuboBuilder(
        num_variables, [f"binary_{index}" for index in range(num_variables)]
    )
    total_pairs = num_variables * (num_variables - 1) // 2
    total_work = num_variables + total_pairs
    report_every = max(1, total_work // 1000)
    next_report = report_every
    if progress:
        progress(0, total_work, "linear terms")
    for variable in range(num_variables):
        if rng.random() >= parameters["zero_linear_probability"]:
            builder.add_linear(variable, _coefficient(parameters, rng))
        if progress and variable + 1 >= next_report:
            progress(variable + 1, total_work, "linear terms")
            next_report += report_every
    for first, second in _sample_quadratic_pairs(
        num_variables, parameters["quadratic_density"], rng
    ):
        builder.add_quadratic(first, second, _coefficient(parameters, rng))
        if progress:
            scanned = _pair_prefix(num_variables, first) + (second - first)
            completed = num_variables + scanned
            if completed >= next_report:
                progress(completed, total_work, "quadratic terms")
                next_report = completed + report_every

    if progress:
        progress(max(0, total_work - 1), total_work, "validating")

    return {
        "qubo": builder.build(),
        "problem_data": {
            "description": "No domain data; coefficients are generated directly."
        },
        "encoding": {
            "description": "Variables are generic binary values.",
            "energy_relationship": "The saved coefficients directly define the objective.",
        },
    }


def validate(problem: dict) -> dict:
    from .validation import add_warning, finite_statistics, validation_result

    result = validation_result()
    qubo = problem["qubo"]
    num_variables = qubo["num_variables"]
    linear_by_variable = {index: coefficient for index, coefficient in qubo["linear"]}
    quadratic = qubo["quadratic"]
    active_quadratic = set()
    for first, second, _ in quadratic:
        active_quadratic.update((first, second))
    inactive = [
        variable
        for variable in range(num_variables)
        if variable not in active_quadratic and variable not in linear_by_variable
    ]
    coefficients = [coefficient for _, coefficient in qubo["linear"]]
    coefficients += [coefficient for _, _, coefficient in quadratic]
    positive = sum(coefficient > 0 for coefficient in coefficients)
    negative = sum(coefficient < 0 for coefficient in coefficients)
    zero = sum(coefficient == 0 for coefficient in coefficients)
    if not quadratic and num_variables > 1:
        add_warning(result, "no_quadratic_interactions", "Random QUBO contains no quadratic interactions.")
    if inactive:
        add_warning(result, "inactive_random_variables", f"Random QUBO has {len(inactive)} inactive variables.")
    if coefficients and (positive == 0 or negative == 0):
        add_warning(result, "one_sided_coefficients", "All nonzero QUBO coefficients have the same sign.")
    possible_pairs = num_variables * (num_variables - 1) // 2
    result["characteristics"].update(
        {
            "realized_quadratic_density": len(quadratic) / possible_pairs if possible_pairs else 0.0,
            "requested_quadratic_density": problem["parameters"]["quadratic_density"],
            "positive_coefficient_fraction": positive / len(coefficients) if coefficients else None,
            "negative_coefficient_fraction": negative / len(coefficients) if coefficients else None,
            "stored_zero_coefficients": zero,
            "inactive_random_variables": len(inactive),
            "random_coefficient_statistics": finite_statistics(coefficients),
            "encoding_check": {
                "method": "direct_definition",
                "samples_checked": 0,
                "maximum_absolute_error": 0.0,
            },
        }
    )
    return result
