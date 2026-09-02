"""QUBO solver adapters and their friendly manager."""

from .manager import (
    SolverCapabilityError,
    describe_solver,
    list_solvers,
    save_result,
    solve_problem,
)

__all__ = [
    "SolverCapabilityError",
    "describe_solver",
    "list_solvers",
    "save_result",
    "solve_problem",
]
