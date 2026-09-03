# QUBO Solvers

The project currently provides five solver types. Every solver reads the same sparse
QUBO representation and returns the same result structure. The returned binary sample
is evaluated again with the project's independent `qubo_energy` function before a result
is saved.

Use the friendly command line to see the installed choices and defaults:

```bash
python solve.py --list
python solve.py --describe exact
python solve.py --describe scipy_highs
python solve.py --describe ocean_sa
python solve.py --describe simulated_bifurcation
python solve.py --describe transverse_route
```

## Choosing a solver

| Solver type | Result guarantee | Best use | GPU |
|---|---|---|---|
| `exact` | Proven global optimum | Ground truth for very small QUBOs | No |
| `scipy_highs` | Proven optimum when status is `optimal`; otherwise a feasible incumbent | Small or moderately sparse QUBOs | No |
| `ocean_sa` | Heuristic solution | Broad CPU baseline, including sparse instances | No |
| `simulated_bifurcation` | Heuristic solution | Dense quadratic models and highly parallel searches | CUDA |
| `transverse_route` | Heuristic solution | Batched geometric escape dynamics on dense or sparse QUBOs | CUDA |

Exact enumeration grows as \(2^n\). HiGHS linearization adds one product variable and
three constraints per stored quadratic interaction. Simulated bifurcation constructs a
dense \(n \times n\) matrix even when the saved QUBO is sparse. These three facts are more
important than variable count alone when selecting a solver.

## Progress snapshots

Every solver accepts `num_snapshots`, an integer that defaults to `0`. A positive value
stores that many best-incumbent checkpoints at equal fractions of the solver's natural
work budget. The final checkpoint is always at progress `1.0` and matches the returned
solution. Each available checkpoint contains its binary sample, independently recomputed
energy, verification details, work target, and (when measurable) elapsed wall time.

The progress unit depends on the solver:

| Solver | Snapshot progress unit | Collection method |
|---|---|---|
| Exact Enumeration | Enumerated states | Captured during one enumeration |
| SciPy + HiGHS | Time limit, or node limit when time is unlimited | Buffered live-incumbent callbacks during one solve |
| Ocean SA | Completed reads | Reconstructed from one sampler response |
| Simulated Bifurcation | Evolution-step limit | Deterministic checkpoint replays |
| Transverse-Route Flow | Flow steps | Captured during one batched integration |

Discrete work budgets round checkpoint targets upward. Consequently, `num_snapshots`
cannot exceed the number of states, reads, or evolution steps available to the relevant
solver. HiGHS can return a checkpoint with `solution: null` when it has not found an
incumbent by that point.

SciPy's public `milp` interface does not expose live incumbents, so the SciPy + HiGHS
adapter passes the SciPy sparse model to the native `highspy` interface. Callback data is
buffered in memory and written only after the single solve finishes. Simulated-bifurcation
still uses deterministic checkpoint replays because its public optimizer interface does
not expose live state.

## Exact Enumeration (`exact`)

### What it does

Exact Enumeration evaluates every one of the \(2^n\) binary assignments. Evaluations are
performed in batches, so memory use stays bounded, but runtime remains exponential. A
completed run has status `optimal` and is suitable as benchmark ground truth.

The implementation is built into this project and uses NumPy only. It is intentionally
restricted to the CPU because its useful range is small and a separate GPU implementation
would introduce a second algorithmic baseline.

### Parameters

| Parameter | Type | Default | Meaning for Exact Enumeration |
|---|---:|---:|---|
| `max_variables` | integer | `24` | Refuses instances above this size before attempting \(2^n\) evaluations. Raising it does not make the method more scalable. |
| `batch_size` | integer | `65536` | Number of assignments represented and evaluated together. Higher values generally improve throughput but require more memory. |
| `num_snapshots` | integer | `0` | Number of equally spaced best-so-far state checkpoints. Zero disables snapshots. It cannot exceed (2^n). |

### Suggested sweeps

Do not sweep this solver for solution quality: every completed run must return the same
optimal energy. Use it to establish ground truth at sizes such as 10, 15, 20, and—where
runtime permits—24 variables.

For a one-time performance calibration, try:

```yaml
batch_size: [4096, 16384, 65536, 262144]
```

After choosing a sensible batch size for the machine, keep it fixed. `max_variables` is a
safety gate rather than an experimental hyperparameter.

## SciPy + HiGHS MILP (`scipy_highs`)

### What it does

The adapter uses SciPy sparse matrices and the native HiGHS Python interface. It replaces
every product \(x_i x_j\) with an auxiliary variable \(y_{ij}\) and the exact constraints

```text
y_ij <= x_i
y_ij <= x_j
y_ij >= x_i + x_j - 1
```

The original variables remain binary and each auxiliary variable is bounded between zero
and one. If HiGHS reports `optimal`, global optimality has been proven. If a time or node
limit is reached after an incumbent is found, the saved status is `feasible` and the final
bound and MIP gap are recorded. See the
[SciPy `milp` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

HiGHS is CPU-only in this interface.

### Parameters

| Parameter | Type | Default | Meaning for SciPy + HiGHS |
|---|---:|---:|---|
| `time_limit` | number or null | `60.0` | Maximum HiGHS solve time in seconds. Use `null` for no limit. Model construction time is still included in the saved wall time. |
| `mip_rel_gap` | number | `0.0` | Acceptable relative difference between the best feasible objective and the dual bound. Zero requests an exact proof. |
| `node_limit` | integer or null | `null` | Maximum number of branch-and-bound nodes. `null` leaves it unlimited. |
| `presolve` | boolean | `true` | Enables HiGHS preprocessing that removes or simplifies variables and constraints. |
| `display` | boolean | `false` | Prints the HiGHS progress log to the terminal. It does not change the mathematical model. |
| `max_quadratic_terms` | integer | `25000` | Rejects a QUBO before creating more than this many auxiliary product variables. |
| `max_linearized_variables` | integer | `100000` | Rejects a model when original variables plus product variables exceed this safety limit. |
| `num_snapshots` | integer | `0` | Number of equally spaced incumbent checkpoints. Uses `time_limit` when positive, otherwise a positive `node_limit`; zero disables snapshots. |

### Suggested sweeps

For anytime-performance tests, use a logarithmic time sweep while keeping the exact gap:

```yaml
time_limit: [1, 10, 60, 300, 1800]
mip_rel_gap: 0
```

For quality-versus-proof-effort tests, use:

```yaml
time_limit: 300
mip_rel_gap: [0, 0.0001, 0.001, 0.01]
```

Keep `presolve: true` in the main benchmark. A separate `true`/`false` sweep is useful only
when specifically studying preprocessing. The two maximum-size parameters should normally
remain fixed safety gates.

## D-Wave Ocean Simulated Annealing (`ocean_sa`)

### What it does

This adapter uses `dwave.samplers.SimulatedAnnealingSampler`, the optimized classical
simulated annealer in D-Wave's Ocean software. It runs locally on the CPU; it does not use
a D-Wave quantum processing unit or submit anything to a cloud service. Each read follows
an inverse-temperature schedule and returns one candidate sample. The lowest-energy sample
is saved. See the [Ocean sampler documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html).

### Parameters

| Parameter | Type | Default | Meaning for Ocean Simulated Annealing |
|---|---:|---:|---|
| `num_reads` | integer | `100` | Number of independently initialized annealing trajectories. More reads broaden the search. |
| `num_sweeps` | integer | `1000` | Number of complete variable-update sweeps in each read. More sweeps slow the cooling and increase work per read. |
| `num_sweeps_per_beta` | integer | `1` | Number of sweeps performed before advancing to the next inverse temperature. |
| `beta_schedule_type` | choice | `geometric` | Uses either `geometric` or `linear` interpolation between the beta endpoints. |
| `beta_min` | number or null | `null` | Starting inverse temperature. When both beta values are null, Ocean derives a range from the model biases. |
| `beta_max` | number or null | `null` | Ending inverse temperature. It must be greater than `beta_min` when a manual range is used. |
| `randomize_order` | boolean | `false` | Randomizes variable-update order within sweeps instead of using the sampler's fixed ordering. |
| `proposal_acceptance_criteria` | choice | `Metropolis` | Selects `Metropolis` or `Gibbs` acceptance probabilities for proposed flips. |
| `seed` | integer | `0` | Seeds Ocean's pseudo-random generator. Repeated seeds allow reproducible local runs. |
| `max_variable_updates` | integer | `100000000` | Safety gate on `num_variables * num_reads * num_sweeps`. Raise it deliberately when a larger CPU run is intended. |
| `num_snapshots` | integer | `0` | Number of equally spaced best-so-far read checkpoints. Zero disables snapshots; it cannot exceed `num_reads`. |

### Suggested sweeps

The primary computational budget is approximately `num_reads * num_sweeps`. A compact
quality sweep is:

```yaml
num_reads: [10, 100, 1000]
num_sweeps: [100, 1000, 10000]
```

That full Cartesian product spans a factor of 10,000 in work. For a smaller study, hold
one parameter fixed and sweep the other. Then compare schedule behavior with:

```yaml
beta_schedule_type: [geometric, linear]
randomize_order: [false, true]
```

Use at least 10 distinct seeds for statistical comparisons. Leave `beta_min` and
`beta_max` automatic until there is evidence that schedule tuning is needed; a manual beta
range is strongly dependent on coefficient scale. Keep `max_variable_updates` fixed as a
safety gate rather than treating it as a quality parameter.

## Simulated Bifurcation (`simulated_bifurcation`)

### What it does

This adapter uses the open-source `simulated-bifurcation` package. Multiple oscillator
systems—called agents—are evolved in parallel with PyTorch. The solver supports ballistic,
discrete, heated, and unheated variants. It is heuristic: completion does not prove
optimality. See the
[Simulated Bifurcation optimizer documentation](https://simulated-bifurcation-algorithm.readthedocs.io/en/v2.0.0/main_features/simulated_bifurcation_optimizer.html).

The package uses a dense coefficient matrix and has space complexity proportional to
\(n^2\). A sparse 100,000-variable QUBO therefore remains unsuitable. CUDA can accelerate
large matrix operations and many parallel agents, while small instances may be faster on
the CPU.

### Parameters

| Parameter | Type | Default | Meaning for Simulated Bifurcation |
|---|---:|---:|---|
| `agents` | integer | `128` | Number of independently initialized oscillator systems evolved in parallel. |
| `max_steps` | integer | `10000` | Maximum number of symplectic-Euler evolution steps before returning current candidates. |
| `timeout` | number or null | `60.0` | Optimizer time limit in seconds. `null` leaves step count and convergence as the stopping rules. |
| `mode` | choice | `ballistic` | `ballistic` is usually faster; `discrete` is usually more accurate. |
| `heated` | boolean | `false` | Adds thermal fluctuations to help agents escape local optima. |
| `early_stopping` | boolean | `true` | Freezes an agent after its sampled energy remains unchanged for long enough. |
| `sampling_period` | integer | `50` | Evolution steps between two energy-stability checks. |
| `convergence_threshold` | integer | `50` | Consecutive unchanged checks required to consider an agent converged. |
| `dtype` | choice | `float32` | Uses `float32` or `float64` PyTorch arithmetic. `float64` doubles matrix storage and may be much slower on some GPUs. |
| `device` | string | `cpu` | `cpu`, `auto`, `cuda`, or an indexed device such as `cuda:0`. `auto` selects CUDA when PyTorch reports it available. |
| `seed` | integer | `0` | Seeds PyTorch's randomized agent initialization. CPU and GPU runs are not guaranteed to match. |
| `max_variables` | integer | `5000` | Refuses larger instances before allocating the dense matrix. This is a safety gate, not a claim that every 5,000-variable run is practical. |
| `num_snapshots` | integer | `0` | Number of equally spaced best-solution step checkpoints. Zero disables snapshots; it cannot exceed `max_steps`. |

To use CUDA, install the PyTorch build appropriate for the machine and set:

```yaml
device: cuda
```

An explicit CUDA request fails clearly if CUDA is unavailable. `device: auto` falls back
to the CPU. Exact Enumeration, SciPy + HiGHS, and Ocean SA do not have GPU backends.

### Suggested sweeps

A useful first sweep is:

```yaml
agents: [16, 64, 128, 512]
max_steps: [1000, 5000, 10000]
mode: [ballistic, discrete]
heated: [false, true]
```

This is 48 combinations, so start with `mode: ballistic` and `heated: false`, determine an
appropriate `agents`/`max_steps` budget, and only then compare algorithm variants. Use at
least 10 seeds for quality statistics.

Benchmark `device: cpu` and `device: cuda` as different execution platforms. Do not combine
their raw timings without recording the processor/GPU model, PyTorch version, dtype, and
device; the saved result already records the software version, dtype, and selected device.

## Transverse-Route Geometric Flow (`transverse_route`)

### What it does

This built-in PyTorch solver implements the first-order manifold dynamics specified in
[`new_solver.md`](new_solver.md). It converts the QUBO exactly to Ising form, scales its
fields by the maximum absolute row interaction bound, and evolves independent angular
trajectories in one batch. The longitudinal coordinate is `cos(theta)`, the required route
coordinate is `cos(theta) * sin(theta)`, and the optional higher-order route coordinate is
`sin(theta)^3`. All active interaction channels and agents are concatenated into one dense
GEMM or sparse SpMM per derivative evaluation.

The confinement follows the monotone power schedule from `kappa_initial` to `kappa_final`.
Candidates are obtained from the signs of the longitudinal coordinates and the solver
retains the lowest exact QUBO objective seen at any candidate checkpoint. This is a
heuristic solver and does not claim global optimality.

### Parameters

| Parameter | Type | Default | Meaning for Transverse-Route Flow |
|---|---:|---:|---|
| `agents` | integer | `64` | Independent angular trajectories evolved as matrix columns. |
| `max_steps` | integer | `1000` | Fixed-step integration updates. |
| `time_step` | number | `0.05` | Dimensionless step after coefficient normalization. |
| `mobility` | number | `1.0` | Positive multiplier on the negative energy gradient. |
| `route_strength` | number | `1.0` | Weight of the endpoint-oriented route channel. Keep this at one for the exact collective-flip curvature identity; zero is the route ablation. |
| `gamma` | number | `0.0` | Weight of the optional higher-order midpoint route channel. |
| `kappa_initial` | number | `-1.0` | Initial transverse confinement. Negative values open the transverse manifold. |
| `kappa_final` | number | `2.0` | Final transverse confinement. Positive values favor binary endpoints. |
| `schedule_exponent` | number | `1.0` | Power-law confinement schedule exponent. |
| `integrator` | choice | `euler` | `euler` uses one interaction multiply per step; `heun` uses two for improved stability. |
| `candidate_interval` | integer | `25` | Steps between best-seen discrete candidate evaluations. Snapshot steps are always evaluated too. |
| `matrix_format` | choice | `auto` | `dense`, `sparse`, or automatic selection using matrix density. |
| `sparse_threshold` | number | `0.15` | Directed density at or below which `auto` chooses sparse SpMM. |
| `dtype` | choice | `float32` | `float32` or `float64` PyTorch computation. |
| `device` | string | `cpu` | `cpu`, `auto`, `cuda`, or an indexed CUDA device. |
| `seed` | integer | `0` | Seed for uniform random initial angles. |
| `max_variables` | integer | `100000` | Safety limit for the batched state allocation. |
| `max_dense_variables` | integer | `10000` | Safety gate for dense matrix allocation. |
| `num_snapshots` | integer | `0` | Equally spaced best-seen flow checkpoints. |

The minimal theoretical model uses `route_strength: 1` and `gamma: 0`. To test the full
route-separation model, sweep positive `gamma` values. For direct ablations, compare
`route_strength: 0`, fixed confinement (`kappa_initial == kappa_final`), and the default
continuation schedule. Use `device: auto` or `device: cuda` when enough agents are present
to benefit from accelerator matrix throughput.

## Running one problem

```bash
python solve.py \
  --problem generated_problems/random_qubo/example.json \
  --solver ocean_sa \
  --set num_reads=100 \
  --set num_sweeps=1000 \
  --set num_snapshots=5 \
  --set seed=42
```

On Windows PowerShell, the same command can be written on one line. Run `python solve.py`
without arguments for guided mode.

## Running a configuration

`configs/solve/solve_random_qubo.yaml` is the ready-to-edit Random QUBO solver configuration:

```bash
python solve.py configs/solve/solve_random_qubo.yaml
```

The configuration format is intentionally small:

```yaml
problem_root: ../../generated_problems
problem_files:
  - ../../generated_problems/random_qubo_*/qubos/*.json
problem_limit: 1
continue_on_error: false

execution:
  mode: parallel
  order: problem_first
  cpu_workers: 2

solvers:
  - type: exact
    instance_name: exact_reference
    parameters:
      max_variables: 24

  - type: ocean_sa
    instance_name: ocean_sa_default
    parameters:
      num_reads: 100
      num_sweeps: 1000
      seed: 42

  - type: transverse_route
    instance_name: trf_pool
    devices: [cuda:0, cuda:1, cuda:2, cuda:3]
    replicas: 1
    parameters:
      agents: 64
      max_steps: 1000
```

Every solver entry needs a unique `instance_name`. Each problem must be below a batch's
`qubos` directory. Solutions are placed in its sibling `results` directory, and the solver
instance name is appended to the JSON filename. Paths and wildcard patterns are resolved
relative to the configuration file. Directories
are searched recursively. `problem_limit` is useful for trying a configuration before a
full run. With `continue_on_error: true`, capability-limit failures are printed and the
remaining solver/problem combinations continue.

`execution.mode` is `sequential` or `parallel`. `execution.order` is `problem_first` or
`solver_first` and controls queue priority; parallel completion order depends on runtime.
CPU work uses at most `cpu_workers` processes. A solver's optional `devices` list forms a
GPU pool, with one active task per CUDA device. Fixed `parameters.device: cuda:N`
assignments are also respected and serialized per GPU.

`replicas` defaults to one. A higher value creates independent files named
`INSTANCE_NAME_replica_N`; numeric solver seeds increment for every replica by default.
Set `increment_seed_per_replica: false` to disable that. Replicas are multi-start runs:
one native solver run still executes on one GPU.

## Saved results

Results are stored beside their QUBO batch:

```text
random_qubo_1000_0.01/
├── qubos/
│   ├── manifest.csv
│   └── seed_42.json
└── results/
    ├── manifest.csv
    ├── seed_42--ocean_sa_default.json
    └── seed_42--bifurcation_gpu0.json
```

Each JSON result records the problem ID and source, normalized parameters, package version,
device, status, binary sample, independently recomputed energy, wall time, verification
details, progress snapshots, and solver-specific metrics. `manifest.csv` provides one
compact row per run and records the number of snapshots saved.
