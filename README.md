# QUBO Problem Generator and Solver Benchmarks

A small collection of Python scripts for creating reproducible QUBO problem instances.
Generated problems are saved as ordinary JSON files and can be solved through four simple
solver adapters. A resumable relative-benchmark runner compares solver energies, rankings,
solution distances, and runtimes across a problem set.

## Install

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Easiest use

Start the guided generator and answer the displayed questions:

```bash
python generate.py
```

Press Enter at a question to use its displayed default.

To generate a prepared configuration:

```bash
python generate.py configs/quickstart.yaml
```

To list or inspect available generators:

```bash
python generate.py --list
python generate.py --describe max_cut
```

One problem can also be generated directly:

```bash
python generate.py --problem number_partitioning --set num_numbers=25 --count 3 --seed 42
```

## Configuration files

```yaml
output_folder: generated_problems
base_seed: 42

problems:
  - type: max_cut
    count: 3
    parameters:
      num_vertices: 30
      edge_probability: 0.2
      weighted: true
```

Any omitted parameter uses the default shown by `--describe`. Each instance receives a
different seed. The same problem type, parameters, and seed always produce the same JSON
content and instance ID.

See `configs/all_problem_types.yaml` for a small example of every generator.
See [`PROBLEMS.md`](PROBLEMS.md) for explanations, complete parameter lists, and suggested
benchmark sweeps for every problem family.

## Solving generated problems

List or inspect the available solvers:

```bash
python solve.py --list
python solve.py --describe ocean_sa
```

Run one saved problem:

```bash
python solve.py --problem path/to/problem.json --solver exact
```

Add equally spaced best-solution checkpoints with `num_snapshots` (zero is the default):

```bash
python solve.py --problem path/to/problem.json --solver ocean_sa --set num_snapshots=5
```

Or run the prepared configuration, which selects one small Random QUBO and applies all four
solvers:

```bash
python solve.py configs/solver_quickstart.yaml
```

Running `python solve.py` with no arguments starts a guided mode. Available backends are
Exact Enumeration, SciPy + HiGHS MILP, D-Wave Ocean Simulated Annealing, and Simulated
Bifurcation. Only Simulated Bifurcation supports GPU execution; set its `device` parameter
to `cuda` or `auto`.

See [`SOLVERS.md`](SOLVERS.md) for each solver's limitations, complete parameter list,
GPU instructions, suggested sweeps, configuration format, and saved-result layout.

## Four-solver relative benchmark

Compare SciPy + HiGHS, Ocean SA, and Simulated Bifurcation on CPU and CUDA without using
Exact Enumeration:

```bash
python benchmark_4_solvers.py generated_problems \
  --config configs/relative_4_solver_benchmark.yaml \
  --output BenchmarkResults/relative_4_solver
```

The runner saves progress after every solver attempt, resumes safely, writes a comparison
report for every QUBO, and produces complete-case averages over instances. See
[`docs/RELATIVE_BENCHMARK.md`](docs/RELATIVE_BENCHMARK.md) for setup, metrics, and output
details.

## Available problem families

- Maximum Cut
- Maximum Independent Set
- Maximum Clique
- Graph Coloring
- Number Partitioning
- Quadratic Knapsack
- Set Packing
- SAT / Max-SAT (2-SAT or quadratized 3-SAT)
- Spin Glass
- Random QUBO

Each generator lives in one readable file under `problems/`. Its `PARAMETERS` dictionary
controls what users see in guided mode, and its `generate(parameters, seed)` function
constructs the problem.

## Saved QUBO format

Every JSON file contains generation parameters, original problem data, variable names,
encoding notes, validation metadata, and sparse QUBO coefficients. Saving problems also
creates or updates a flattened `manifest.csv` in the output directory. The convention is:

```text
E(x) = offset
     + sum_i linear[i] * x_i
     + sum_{i<j} quadratic[i,j] * x_i * x_j
```

Linear terms are saved as `[variable, coefficient]`. Quadratic terms are saved as
`[first_variable, second_variable, coefficient]`. This explicitly avoids symmetric-matrix
double-counting ambiguity.

Python code can load an instance and evaluate a sample as follows:

```python
from pathlib import Path

from problems.common import qubo_energy
from problems.manager import load_problem

path = next(Path("generated_problems").glob("max_cut-*.json"))
problem = load_problem(path)
sample = [0] * problem["qubo"]["num_variables"]
energy = qubo_energy(problem["qubo"], sample)
```

## Instance validation

Structural and problem-specific validation runs automatically before an instance is
saved. Small QUBOs are checked exhaustively against an independent objective calculation;
larger QUBOs use deterministic sample checks. Validation errors prevent saving, while
potentially trivial or biased structures are recorded as warnings.

Warning codes can optionally be rejected using a predeclared configuration rule:

```yaml
validation:
  max_attempts: 20
  reject_warnings:
    - disconnected_graph
    - isolated_vertices
```

See [`VALIDATION.md`](VALIDATION.md) for all checks, warning codes, manifest columns, and
guidance on constructing balanced benchmark collections.
