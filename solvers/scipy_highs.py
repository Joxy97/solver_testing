"""QUBO-to-MILP linearization solved by HiGHS through SciPy."""

from __future__ import annotations

import numpy as np

from .common import SolverCapabilityError, package_version, require_package


NAME = "SciPy + HiGHS MILP"
DESCRIPTION = """
Converts each quadratic product to a bounded auxiliary variable with exact linear
constraints, then calls the HiGHS mixed-integer solver through scipy.optimize.milp.
An optimal status is a proof of global optimality; a time-limited incumbent is feasible.
"""

PARAMETERS = {
    "time_limit": {
        "type": "number",
        "default": 60.0,
        "min": 0.0,
        "allow_none": True,
        "description": "Maximum solve time in seconds; null means no time limit",
    },
    "mip_rel_gap": {
        "type": "number",
        "default": 0.0,
        "min": 0.0,
        "description": "Stop when the relative gap between incumbent and bound reaches this value",
    },
    "node_limit": {
        "type": "integer",
        "default": None,
        "min": 0,
        "allow_none": True,
        "description": "Maximum branch-and-bound nodes; null means no node limit",
    },
    "presolve": {
        "type": "boolean",
        "default": True,
        "description": "Allow HiGHS to simplify the MILP before branch-and-bound",
    },
    "display": {
        "type": "boolean",
        "default": False,
        "description": "Print the HiGHS progress log while solving",
    },
    "max_quadratic_terms": {
        "type": "integer",
        "default": 25_000,
        "min": 0,
        "description": "Safety limit on product variables introduced by linearization",
    },
    "max_linearized_variables": {
        "type": "integer",
        "default": 100_000,
        "min": 1,
        "description": "Safety limit on original plus auxiliary MILP variables",
    },
}


def version() -> str:
    return f"SciPy {package_version('scipy')} (bundled HiGHS)"


def solve(qubo: dict, parameters: dict) -> dict:
    require_package("scipy", "scipy", NAME)
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array

    num_variables = int(qubo["num_variables"])
    quadratic_terms = qubo["quadratic"]
    num_products = len(quadratic_terms)
    total_variables = num_variables + num_products
    if num_products > parameters["max_quadratic_terms"]:
        raise SolverCapabilityError(
            f"MILP linearization would introduce {num_products:,} product variables, above "
            f"max_quadratic_terms={parameters['max_quadratic_terms']:,}."
        )
    if total_variables > parameters["max_linearized_variables"]:
        raise SolverCapabilityError(
            f"MILP linearization would contain {total_variables:,} variables, above "
            f"max_linearized_variables={parameters['max_linearized_variables']:,}."
        )

    objective = np.zeros(total_variables, dtype=np.float64)
    for variable, coefficient in qubo["linear"]:
        objective[int(variable)] += float(coefficient)
    for product, (_, _, coefficient) in enumerate(quadratic_terms):
        objective[num_variables + product] = float(coefficient)

    integrality = np.zeros(total_variables, dtype=np.uint8)
    integrality[:num_variables] = 1
    bounds = Bounds(np.zeros(total_variables), np.ones(total_variables))
    constraints = None

    if num_products:
        products = np.arange(num_products, dtype=np.int64)
        first = np.asarray([term[0] for term in quadratic_terms], dtype=np.int64)
        second = np.asarray([term[1] for term in quadratic_terms], dtype=np.int64)
        auxiliary = num_variables + products

        # y <= x_i, y <= x_j, and y >= x_i + x_j - 1 force y = x_i*x_j.
        rows = np.concatenate(
            (
                3 * products,
                3 * products,
                3 * products + 1,
                3 * products + 1,
                3 * products + 2,
                3 * products + 2,
                3 * products + 2,
            )
        )
        columns = np.concatenate(
            (first, auxiliary, second, auxiliary, first, second, auxiliary)
        )
        data = np.concatenate(
            (
                -np.ones(num_products),
                np.ones(num_products),
                -np.ones(num_products),
                np.ones(num_products),
                -np.ones(num_products),
                -np.ones(num_products),
                np.ones(num_products),
            )
        )
        matrix = coo_array(
            (data, (rows, columns)), shape=(3 * num_products, total_variables)
        ).tocsc()
        lower = np.full(3 * num_products, -np.inf)
        upper = np.full(3 * num_products, np.inf)
        upper[0::3] = 0.0
        upper[1::3] = 0.0
        lower[2::3] = -1.0
        constraints = LinearConstraint(matrix, lower, upper)

    options = {
        "disp": parameters["display"],
        "presolve": parameters["presolve"],
        "mip_rel_gap": parameters["mip_rel_gap"],
    }
    if parameters["time_limit"] is not None:
        options["time_limit"] = parameters["time_limit"]
    if parameters["node_limit"] is not None:
        options["node_limit"] = parameters["node_limit"]

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    if result.x is None:
        raise RuntimeError(f"HiGHS returned no feasible solution: {result.message}")

    original_values = np.asarray(result.x[:num_variables])
    sample = np.rint(original_values).astype(np.int8)
    if np.any(np.abs(original_values - sample) > 1e-5):
        raise RuntimeError("HiGHS returned a non-binary incumbent for original QUBO variables")

    offset = float(qubo.get("offset", 0.0))
    metrics = {
        "scipy_status": int(result.status),
        "message": str(result.message),
        "mip_node_count": getattr(result, "mip_node_count", None),
        "mip_gap": getattr(result, "mip_gap", None),
        "mip_dual_bound": (
            float(result.mip_dual_bound) + offset
            if getattr(result, "mip_dual_bound", None) is not None
            else None
        ),
        "original_variables": num_variables,
        "auxiliary_product_variables": num_products,
        "linearization_constraints": 3 * num_products,
        "optimality_proven": int(result.status) == 0,
    }
    return {
        "status": "optimal" if int(result.status) == 0 else "feasible",
        "sample": sample.tolist(),
        "reported_energy": float(result.fun) + offset,
        "device": "cpu",
        "metrics": metrics,
    }
