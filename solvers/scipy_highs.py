"""QUBO-to-MILP linearization solved by HiGHS with live snapshots."""

from __future__ import annotations

import time

import numpy as np

from .common import (
    SolverCapabilityError,
    equally_spaced_targets,
    package_version,
    require_package,
    snapshot_record,
)


NAME = "SciPy + HiGHS MILP"
DESCRIPTION = """
Converts each quadratic product to a bounded auxiliary variable with exact linear
constraints, then calls HiGHS. Native HiGHS callbacks buffer incumbent snapshots during
one solve, so enabling snapshots does not repeat the optimization. An optimal status is
a proof of global optimality; a time-limited incumbent is feasible.
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
    "num_snapshots": {
        "type": "integer",
        "default": 0,
        "min": 0,
        "description": "Equally spaced incumbent checkpoints; zero disables snapshots",
    },
}


def version() -> str:
    return f"SciPy {package_version('scipy')}; highspy {package_version('highspy')}"


def solve(qubo: dict, parameters: dict) -> dict:
    require_package("scipy", "scipy", NAME)
    require_package("highspy", "highspy", NAME)
    import highspy
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

    matrix = None
    row_lower = np.empty(0, dtype=np.float64)
    row_upper = np.empty(0, dtype=np.float64)

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
        row_lower = np.full(3 * num_products, -np.inf)
        row_upper = np.full(3 * num_products, np.inf)
        row_upper[0::3] = 0.0
        row_upper[1::3] = 0.0
        row_lower[2::3] = -1.0

    def extract_sample(values):
        if values is None or len(values) < num_variables:
            return None
        original_values = np.asarray(values[:num_variables])
        checkpoint_sample = np.rint(original_values).astype(np.int8)
        if np.any(np.abs(original_values - checkpoint_sample) > 1e-5):
            raise RuntimeError(
                "HiGHS returned a non-binary incumbent for original QUBO variables"
            )
        return checkpoint_sample.tolist()

    offset = float(qubo.get("offset", 0.0))
    num_snapshots = parameters["num_snapshots"]
    snapshots = []
    snapshot_unit = None
    snapshot_total = None
    targets = []
    if num_snapshots:
        time_limit = parameters["time_limit"]
        node_limit = parameters["node_limit"]
        if time_limit is not None and time_limit > 0:
            snapshot_unit = "seconds"
            snapshot_total = float(time_limit)
            targets = [
                snapshot_total * index / num_snapshots
                for index in range(1, num_snapshots + 1)
            ]
        elif node_limit is not None and node_limit > 0:
            snapshot_unit = "branch_and_bound_nodes"
            snapshot_total = int(node_limit)
            targets = equally_spaced_targets(snapshot_total, num_snapshots)
        else:
            raise ValueError(
                "SciPy + HiGHS snapshots require a positive time_limit or node_limit"
            )

    lp = highspy.HighsLp()
    lp.num_col_ = total_variables
    lp.num_row_ = 3 * num_products
    lp.col_cost_ = objective
    lp.col_lower_ = np.zeros(total_variables, dtype=np.float64)
    lp.col_upper_ = np.ones(total_variables, dtype=np.float64)
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.offset_ = offset
    lp.integrality_ = (
        [highspy.HighsVarType.kInteger] * num_variables
        + [highspy.HighsVarType.kContinuous] * num_products
    )
    if matrix is not None:
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = total_variables
        lp.a_matrix_.num_row_ = 3 * num_products
        lp.a_matrix_.start_ = matrix.indptr
        lp.a_matrix_.index_ = matrix.indices
        lp.a_matrix_.value_ = matrix.data

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", parameters["display"])
    highs.setOptionValue("presolve", "on" if parameters["presolve"] else "off")
    highs.setOptionValue("mip_rel_gap", parameters["mip_rel_gap"])
    if parameters["time_limit"] is not None:
        highs.setOptionValue("time_limit", parameters["time_limit"])
    if parameters["node_limit"] is not None:
        highs.setOptionValue("mip_max_nodes", parameters["node_limit"])
    if highs.passModel(lp) == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS rejected the linearized QUBO model")

    # Every unconstrained QUBO has this feasible point. Supplying it prevents a short,
    # difficult MILP run from reaching its limit without any incumbent to return.
    initial_values = np.zeros(total_variables, dtype=np.float64)
    initial_indices = np.arange(total_variables, dtype=np.int32)
    if (
        highs.setSolution(total_variables, initial_indices, initial_values)
        == highspy.HighsStatus.kError
    ):
        raise RuntimeError("HiGHS rejected the all-zero initial incumbent")

    best_values = initial_values
    best_energy = offset
    next_snapshot = 0

    def add_snapshot(target, observed, node_count, gap, dual_bound):
        nonlocal next_snapshot
        sample_at_target = extract_sample(best_values)
        snapshots.append(
            snapshot_record(
                index=next_snapshot + 1,
                count=num_snapshots,
                unit=snapshot_unit,
                target=target,
                total=snapshot_total,
                sample=sample_at_target,
                reported_energy=best_energy if sample_at_target is not None else None,
                elapsed_seconds=observed,
                metrics={
                    "mip_node_count": int(node_count),
                    "mip_gap": float(gap),
                    "mip_dual_bound": float(dual_bound),
                },
            )
        )
        next_snapshot += 1

    def callback(callback_type, _message, data_out, _data_in, _user_data):
        nonlocal best_values, best_energy
        observed = float(data_out.running_time)
        progress = observed if snapshot_unit == "seconds" else int(data_out.mip_node_count)
        # Reserve the last checkpoint for the final solution returned by HiGHS so the
        # normalized result always ends with exactly the solver's final incumbent.
        while next_snapshot < len(targets) - 1 and targets[next_snapshot] <= progress:
            add_snapshot(
                targets[next_snapshot],
                observed,
                data_out.mip_node_count,
                data_out.mip_gap,
                data_out.mip_dual_bound,
            )
        if callback_type == highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution:
            candidate = np.asarray(data_out.mip_solution, dtype=np.float64)
            if len(candidate) >= total_variables:
                best_values = candidate.copy()
                best_energy = float(np.dot(objective, candidate) + offset)

    if num_snapshots:
        highs.setCallback(callback, None)
        highs.startCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)
        highs.startCallback(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)

    solve_started = time.perf_counter()
    run_status = highs.run()
    solve_elapsed = time.perf_counter() - solve_started
    model_status = highs.getModelStatus()
    info = highs.getInfo()
    solution = highs.getSolution()
    sample = extract_sample(solution.col_value if solution.value_valid else None)
    if sample is None:
        raise RuntimeError(
            f"HiGHS returned no feasible solution: {highs.modelStatusToString(model_status)}"
        )
    final_values = np.asarray(solution.col_value, dtype=np.float64)
    final_energy = float(np.dot(objective, final_values) + offset)

    best_values = final_values
    best_energy = final_energy
    while next_snapshot < len(targets):
        add_snapshot(
            targets[next_snapshot],
            solve_elapsed,
            info.mip_node_count,
            info.mip_gap,
            info.mip_dual_bound,
        )

    optimal = model_status == highspy.HighsModelStatus.kOptimal
    if run_status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS failed: {highs.modelStatusToString(model_status)}")
    metrics = {
        "highs_model_status": int(model_status),
        "message": highs.modelStatusToString(model_status),
        "mip_node_count": info.mip_node_count,
        "mip_gap": info.mip_gap,
        "mip_dual_bound": info.mip_dual_bound,
        "original_variables": num_variables,
        "auxiliary_product_variables": num_products,
        "linearization_constraints": 3 * num_products,
        "optimality_proven": optimal,
    }
    return {
        "status": "optimal" if optimal else "feasible",
        "sample": sample,
        "reported_energy": final_energy,
        "device": "cpu",
        "metrics": metrics,
        "snapshots": snapshots,
    }
