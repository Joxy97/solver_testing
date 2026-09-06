"""Small shared helpers used by the problem-generator scripts."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable

import networkx as nx
import numpy as np

from .storage import (
    BLOCK, HEADER, PACKED, TOLERANCE, MappedQubo, TermAccumulator,
    _header, linear_values, pair_start, quadratic_blocks, write_sorted,
)


QUBO_CONVENTION = (
    "E(x) = offset + sum_i linear[i]*x_i + "
    "sum_{i<j} quadratic[i,j]*x_i*x_j"
)


class QuboBuilder:
    """Accumulate coefficients on disk, using bounded buffers for sparse terms."""

    def __init__(self, num_variables, variable_names, *, packed=False):
        self.num_variables, self.variable_names = num_variables, variable_names
        self.offset = 0.0
        self._temporary = tempfile.TemporaryDirectory(prefix="qubo-build-")
        self.path = Path(self._temporary.name) / "data.qubo"
        self.packed = packed
        self.quadratic = None
        self.linear = None
        if packed:
            pairs = num_variables * (num_variables - 1) // 2
            with self.path.open("wb") as handle:
                handle.truncate(HEADER.size + 8 * (num_variables + pairs))
            self.linear = np.memmap(self.path, mode="r+", dtype="<f8", offset=HEADER.size, shape=(num_variables,))
            self.quadratic = (np.memmap(self.path, mode="r+", dtype="<f8", offset=HEADER.size + 8 * num_variables,
                                        shape=(pairs,)) if pairs else np.empty(0, dtype="<f8"))
        else:
            self.linear = np.lib.format.open_memmap(Path(self._temporary.name) / "linear.npy",
                                                    mode="w+", dtype="<f8", shape=(num_variables,))
            self.quadratic = TermAccumulator()

    def add_linear(self, variable: int, coefficient: float) -> None:
        self._check_variable(variable)
        self.linear[variable] += float(coefficient)

    def add_quadratic(self, first: int, second: int, coefficient: float) -> None:
        self._check_variable(first)
        self._check_variable(second)
        if first == second:
            self.add_linear(first, coefficient)
            return
        first, second = min(first, second), max(first, second)
        if self.packed:
            self.quadratic[pair_start(self.num_variables, first) + second - first - 1] += float(coefficient)
        else:
            self.quadratic.add(first, second, float(coefficient))

    def add_offset(self, value: float) -> None:
        self.offset += float(value)

    def add_square(
        self,
        coefficients: dict[int, float],
        constant: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        """Add weight * (constant + sum(coefficients[i] * x_i))**2."""
        self.add_offset(weight * constant * constant)
        items = list(coefficients.items())
        for variable, coefficient in items:
            self.add_linear(
                variable,
                weight * (coefficient * coefficient + 2.0 * constant * coefficient),
            )
        for position, (first, first_coefficient) in enumerate(items):
            if self.packed:
                for start in range(position + 1, len(items), BLOCK):
                    block = items[start : start + BLOCK]
                    seconds = np.fromiter((item[0] for item in block), dtype=np.int64)
                    values = np.fromiter((item[1] for item in block), dtype=np.float64)
                    rows, columns = np.minimum(first, seconds), np.maximum(first, seconds)
                    positions = rows * (2 * self.num_variables - rows - 1) // 2 + columns - rows - 1
                    self.quadratic[positions] += weight * 2.0 * first_coefficient * values
                continue
            for next_position in range(position + 1, len(items)):
                second, second_coefficient = items[next_position]
                self.add_quadratic(
                    first,
                    second,
                    weight * 2.0 * first_coefficient * second_coefficient,
                )

    def add_affine_product(
        self,
        left: tuple[float, dict[int, float]],
        right: tuple[float, dict[int, float]],
        weight: float = 1.0,
    ) -> None:
        """Add weight times the product of two affine binary expressions."""
        left_constant, left_terms = left
        right_constant, right_terms = right
        self.add_offset(weight * left_constant * right_constant)
        for variable, coefficient in left_terms.items():
            self.add_linear(variable, weight * coefficient * right_constant)
        for variable, coefficient in right_terms.items():
            self.add_linear(variable, weight * coefficient * left_constant)
        for first, first_coefficient in left_terms.items():
            for second, second_coefficient in right_terms.items():
                if first == second:
                    self.add_linear(first, weight * first_coefficient * second_coefficient)
                else:
                    self.add_quadratic(
                        first,
                        second,
                        weight * first_coefficient * second_coefficient,
                    )

    def build(self, sorted_terms=None) -> dict:
        for start in range(0, self.num_variables, BLOCK):
            block = self.linear[start : start + BLOCK]
            block[np.abs(block) <= TOLERANCE] = 0
        if self.packed:
            count = linear_count = 0
            for array in (self.linear, self.quadratic):
                nonzero = 0
                for start in range(0, len(array), BLOCK):
                    block = array[start : start + BLOCK]
                    block[np.abs(block) <= TOLERANCE] = 0
                    nonzero += int(np.count_nonzero(block))
                if array is self.linear:
                    linear_count = nonzero
                else:
                    count = nonzero
            with self.path.open("r+b") as handle:
                handle.write(_header(self.num_variables, count, linear_count, self.offset, PACKED))
        else:
            terms = self.quadratic.terms() if sorted_terms is None else sorted_terms
            write_sorted(self.path, self.num_variables, self.linear, terms, self.offset)
        self._close_arrays()
        result = MappedQubo(self.path, temporary=self._temporary)
        self._temporary = None
        result["variable_names"] = self.variable_names
        return result

    def _close_arrays(self):
        for value in (self.linear, self.quadratic):
            if isinstance(value, np.memmap):
                value.flush()
                value._mmap.close()
            elif isinstance(value, TermAccumulator):
                value.close()
        self.linear = self.quadratic = None

    def __del__(self):
        self._close_arrays()
        if self._temporary is not None:
            self._temporary.cleanup()

    def _check_variable(self, variable: int) -> None:
        if not 0 <= variable < self.num_variables:
            raise IndexError(f"Variable {variable} is outside 0..{self.num_variables - 1}")


def qubo_energy(qubo: dict, sample: Iterable[int]) -> float:
    """Evaluate a binary sample using the project's saved QUBO convention."""
    values = np.asarray(sample) if isinstance(sample, (list, tuple, np.ndarray)) else np.fromiter(sample, dtype=np.float64)
    expected = qubo["num_variables"]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} binary values, received {len(values)}")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("QUBO samples must contain only 0 and 1")
    energy = float(qubo.get("offset", 0.0))
    linear = linear_values(qubo)
    for start in range(0, expected, BLOCK):
        energy += float(np.dot(linear[start : start + BLOCK], values[start : start + BLOCK]))
    for first, second, coefficients in quadratic_blocks(qubo):
        energy += float(np.dot(coefficients, values[first] * values[second]))
    return float(energy)


def graph_parameter_schema() -> dict:
    return {
        "num_vertices": {
            "type": "integer",
            "default": 20,
            "min": 2,
            "description": "Number of vertices in the graph",
        },
        "graph_type": {
            "type": "choice",
            "default": "erdos_renyi",
            "choices": ["erdos_renyi", "random_regular", "cycle", "complete"],
            "description": "Graph generation method",
        },
        "edge_probability": {
            "type": "number",
            "default": 0.25,
            "min": 0.0,
            "max": 1.0,
            "description": "Edge probability for an Erdos-Renyi graph",
        },
        "degree": {
            "type": "integer",
            "default": 3,
            "min": 1,
            "description": "Vertex degree for a random-regular graph",
        },
        "weighted": {
            "type": "boolean",
            "default": False,
            "description": "Whether graph edges receive random integer weights",
        },
        "min_edge_weight": {
            "type": "integer",
            "default": 1,
            "min": 1,
            "description": "Smallest random edge weight when weighted is enabled",
        },
        "max_edge_weight": {
            "type": "integer",
            "default": 10,
            "min": 1,
            "description": "Largest random edge weight when weighted is enabled",
        },
    }


def generate_graph(parameters: dict, rng: np.random.Generator) -> list[list[float]]:
    """Return sorted [u, v, weight] records for a generated undirected graph."""
    num_vertices = parameters["num_vertices"]
    graph_type = parameters["graph_type"]
    graph_seed = int(rng.integers(0, 2**32 - 1))

    if graph_type == "erdos_renyi":
        graph = nx.gnp_random_graph(
            num_vertices,
            parameters["edge_probability"],
            seed=graph_seed,
        )
    elif graph_type == "random_regular":
        degree = parameters["degree"]
        if degree >= num_vertices:
            raise ValueError("degree must be smaller than num_vertices")
        if degree * num_vertices % 2:
            raise ValueError("degree * num_vertices must be even for a regular graph")
        graph = nx.random_regular_graph(degree, num_vertices, seed=graph_seed)
    elif graph_type == "cycle":
        graph = nx.cycle_graph(num_vertices)
    elif graph_type == "complete":
        graph = nx.complete_graph(num_vertices)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    minimum = parameters["min_edge_weight"]
    maximum = parameters["max_edge_weight"]
    if minimum > maximum:
        raise ValueError("min_edge_weight cannot be larger than max_edge_weight")

    edges = []
    for first, second in sorted(tuple(sorted(edge)) for edge in graph.edges()):
        weight = int(rng.integers(minimum, maximum + 1)) if parameters["weighted"] else 1
        edges.append([int(first), int(second), weight])
    return edges


def random_positive_integers(
    rng: np.random.Generator, count: int, minimum: int, maximum: int
) -> list[int]:
    if minimum < 1 or minimum > maximum:
        raise ValueError("Integer range must satisfy 1 <= minimum <= maximum")
    return [int(value) for value in rng.integers(minimum, maximum + 1, size=count)]


def slack_weights(maximum_value: int) -> list[int]:
    """Binary-like weights that represent every integer from 0 to maximum_value."""
    if maximum_value < 0:
        raise ValueError("maximum_value must be non-negative")
    weights: list[int] = []
    represented = 0
    power = 1
    while represented < maximum_value:
        weight = min(power, maximum_value - represented)
        weights.append(weight)
        represented += weight
        power *= 2
    return weights
