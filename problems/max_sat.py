"""Weighted Max-2-SAT and Max-3-SAT QUBO generator."""

import numpy as np

from .common import QuboBuilder


NAME = "SAT / Max-SAT"
DESCRIPTION = "Generate weighted 2-SAT or 3-SAT clauses and minimize unsatisfied clause weight."
PARAMETERS = {
    "num_variables": {
        "type": "integer",
        "default": 20,
        "min": 2,
        "description": "Number of Boolean variables",
    },
    "num_clauses": {
        "type": "integer",
        "default": 80,
        "min": 1,
        "description": "Number of clauses",
    },
    "clause_size": {
        "type": "choice",
        "default": 3,
        "choices": [2, 3],
        "description": "Number of literals per clause",
    },
    "weighted_clauses": {
        "type": "boolean",
        "default": False,
        "description": "Assign random integer weights to clauses",
    },
    "min_clause_weight": {
        "type": "integer",
        "default": 1,
        "min": 1,
        "description": "Smallest clause weight",
    },
    "max_clause_weight": {
        "type": "integer",
        "default": 10,
        "min": 1,
        "description": "Largest clause weight",
    },
    "planted_solution": {
        "type": "boolean",
        "default": True,
        "description": "Generate clauses satisfied by a hidden assignment",
    },
    "ancilla_penalty": {
        "type": "number",
        "default": None,
        "allow_none": True,
        "min": 0.0,
        "description": "3-SAT product penalty; blank selects a safe value",
    },
}


def _false_indicator(literal: dict) -> tuple[float, dict[int, float]]:
    variable = literal["variable"]
    if literal["negated"]:
        return 0.0, {variable: 1.0}
    return 1.0, {variable: -1.0}


def generate(parameters: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    num_variables = parameters["num_variables"]
    clause_size = parameters["clause_size"]
    if num_variables < clause_size:
        raise ValueError("num_variables must be at least clause_size")
    if parameters["min_clause_weight"] > parameters["max_clause_weight"]:
        raise ValueError("min_clause_weight cannot exceed max_clause_weight")

    planted = (
        [int(value) for value in rng.integers(0, 2, size=num_variables)]
        if parameters["planted_solution"]
        else None
    )
    clauses = []
    for _ in range(parameters["num_clauses"]):
        variables = [
            int(value) for value in rng.choice(num_variables, size=clause_size, replace=False)
        ]
        negated = [bool(value) for value in rng.integers(0, 2, size=clause_size)]
        literals = [
            {"variable": variable, "negated": is_negated}
            for variable, is_negated in zip(variables, negated)
        ]
        if planted is not None:
            satisfied = any(
                (planted[literal["variable"]] == 0)
                if literal["negated"]
                else (planted[literal["variable"]] == 1)
                for literal in literals
            )
            if not satisfied:
                selected = int(rng.integers(0, clause_size))
                variable = literals[selected]["variable"]
                literals[selected]["negated"] = planted[variable] == 0
        weight = (
            int(rng.integers(parameters["min_clause_weight"], parameters["max_clause_weight"] + 1))
            if parameters["weighted_clauses"]
            else 1
        )
        clauses.append({"literals": literals, "weight": weight})

    num_ancillas = len(clauses) if clause_size == 3 else 0
    variable_names = [f"boolean_{index}" for index in range(num_variables)]
    variable_names += [f"clause_{index}_ancilla" for index in range(num_ancillas)]
    builder = QuboBuilder(num_variables + num_ancillas, variable_names)
    ancilla_records = []
    configured_penalty = parameters["ancilla_penalty"]

    for clause_index, clause in enumerate(clauses):
        false_terms = [_false_indicator(literal) for literal in clause["literals"]]
        weight = float(clause["weight"])
        if clause_size == 2:
            builder.add_affine_product(false_terms[0], false_terms[1], weight)
            continue

        ancilla = num_variables + clause_index
        ancilla_expression = (0.0, {ancilla: 1.0})
        penalty = configured_penalty
        if penalty is None:
            penalty = weight + 1.0

        # z = y1*y2, where each y is a literal's false indicator.
        builder.add_affine_product(false_terms[0], false_terms[1], penalty)
        builder.add_affine_product(false_terms[0], ancilla_expression, -2.0 * penalty)
        builder.add_affine_product(false_terms[1], ancilla_expression, -2.0 * penalty)
        builder.add_linear(ancilla, 3.0 * penalty)
        builder.add_affine_product(ancilla_expression, false_terms[2], weight)
        ancilla_records.append(
            {
                "clause": clause_index,
                "variable": ancilla,
                "represents": "first two literals are both false",
                "penalty": penalty,
            }
        )

    resolved = dict(parameters)
    if clause_size == 3 and configured_penalty is None:
        resolved["ancilla_penalty"] = "per_clause_weight_plus_one"
    return {
        "parameters": resolved,
        "qubo": builder.build(),
        "problem_data": {
            "num_boolean_variables": num_variables,
            "clauses": clauses,
            "planted_assignment": planted,
            "ancillas": ancilla_records,
        },
        "encoding": {
            "description": (
                "Primary variables are Boolean assignments; each 3-SAT clause has one product ancilla."
            ),
            "energy_relationship": (
                "With consistent ancillas, energy equals total unsatisfied clause weight."
            ),
        },
    }


def validate(problem: dict) -> dict:
    import itertools

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
    num_variables = data["num_boolean_variables"]
    clauses = data["clauses"]
    clause_size = parameters["clause_size"]
    canonical_clauses = []
    occurrences = [0] * num_variables
    positive_literals = 0
    total_literals = 0
    for clause_index, clause in enumerate(clauses):
        literals = clause["literals"]
        if len(literals) != clause_size or clause["weight"] <= 0:
            add_error(result, "invalid_clause", f"Clause {clause_index} has invalid size or weight.")
            continue
        variables = [literal["variable"] for literal in literals]
        if len(set(variables)) != len(variables):
            add_error(result, "repeated_clause_variable", f"Clause {clause_index} repeats a variable.")
        if any(not isinstance(variable, int) or not 0 <= variable < num_variables for variable in variables):
            add_error(result, "literal_out_of_range", f"Clause {clause_index} contains an invalid variable.")
            continue
        canonical = tuple(sorted((literal["variable"], bool(literal["negated"])) for literal in literals))
        canonical_clauses.append(canonical)
        for literal in literals:
            occurrences[literal["variable"]] += 1
            positive_literals += int(not literal["negated"])
            total_literals += 1

    duplicate_clauses = len(canonical_clauses) - len(set(canonical_clauses))
    unused_variables = sum(count == 0 for count in occurrences)
    if duplicate_clauses:
        add_warning(result, "duplicate_clauses", f"The formula contains {duplicate_clauses} duplicate clauses.")
    if unused_variables:
        add_warning(result, "unused_boolean_variables", f"The formula contains {unused_variables} unused variables.")
    positive_fraction = positive_literals / total_literals if total_literals else 0.0
    if total_literals and (positive_fraction < 0.2 or positive_fraction > 0.8):
        add_warning(result, "literal_sign_imbalance", "Positive and negative literal frequencies are strongly imbalanced.")

    planted = data["planted_assignment"]
    if planted is not None:
        if len(planted) != num_variables or any(value not in (0, 1) for value in planted):
            add_error(result, "invalid_planted_assignment", "Planted assignment is malformed.")
        elif _unsatisfied_weight(clauses, planted):
            add_error(result, "broken_planted_assignment", "Planted assignment does not satisfy every clause.")

    ancillas = data["ancillas"]
    if clause_size == 3:
        if len(ancillas) != len(clauses):
            add_error(result, "invalid_ancilla_count", "3-SAT requires one ancilla per clause.")
        for record, clause in zip(ancillas, clauses):
            if record["penalty"] <= clause["weight"]:
                add_warning(
                    result,
                    "weak_ancilla_penalty",
                    "An ancilla penalty does not exceed its clause weight.",
                )
                break
    elif ancillas:
        add_error(result, "unexpected_ancillas", "2-SAT should not contain product ancillas.")

    satisfiable = None
    if num_variables <= 16:
        satisfiable = any(
            _unsatisfied_weight(clauses, assignment) == 0
            for assignment in itertools.product((0, 1), repeat=num_variables)
        )
    result["characteristics"].update(
        {
            "boolean_variables": num_variables,
            "clauses": len(clauses),
            "clause_to_variable_ratio": len(clauses) / num_variables,
            "clause_weight_statistics": finite_statistics(clause["weight"] for clause in clauses),
            "variable_occurrence_statistics": finite_statistics(occurrences),
            "positive_literal_fraction": positive_fraction,
            "duplicate_clauses": duplicate_clauses,
            "unused_boolean_variables": unused_variables,
            "ancilla_variables": len(ancillas),
            "satisfiable_when_checked": satisfiable,
            "has_planted_solution": planted is not None,
        }
    )

    def expected_energy(sample: list[int]) -> float:
        primary = sample[:num_variables]
        if clause_size == 2:
            return float(_unsatisfied_weight(clauses, primary))
        energy = 0.0
        for clause_index, (clause, record) in enumerate(zip(clauses, ancillas)):
            false_values = [
                int(_literal_is_false(literal, primary)) for literal in clause["literals"]
            ]
            ancilla = sample[num_variables + clause_index]
            penalty = record["penalty"]
            energy += penalty * (
                false_values[0] * false_values[1]
                - 2 * false_values[0] * ancilla
                - 2 * false_values[1] * ancilla
                + 3 * ancilla
            )
            energy += clause["weight"] * ancilla * false_values[2]
        return energy

    merge_validation(result, check_encoding(problem, expected_energy))
    return result


def _literal_is_false(literal: dict, assignment) -> bool:
    value = assignment[literal["variable"]]
    return value == 1 if literal["negated"] else value == 0


def _unsatisfied_weight(clauses: list[dict], assignment) -> int:
    return sum(
        clause["weight"]
        for clause in clauses
        if all(_literal_is_false(literal, assignment) for literal in clause["literals"])
    )
