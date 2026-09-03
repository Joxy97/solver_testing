# Six-configuration relative benchmark

`benchmark_6_solvers.py` compares six non-exact solver configurations on every input
QUBO:

1. SciPy + HiGHS
2. Ocean simulated annealing on CPU
3. Simulated bifurcation on CPU
4. Simulated bifurcation on CUDA
5. Transverse-route geometric flow on CPU
6. Transverse-route geometric flow on CUDA

Exact enumeration is deliberately excluded. CPU and CUDA runs share their algorithm's
defaults and seed but are treated as independent benchmark entries.

## Remote setup

Install this project's dependencies and ensure PyTorch is a CUDA-enabled build suitable
for the remote machine. Confirm CUDA before starting the benchmark:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The six-way preflight refuses to start if CUDA or a required solver package is missing.
Use `--allow-missing-solvers` only when partial results are intentionally acceptable.

## Run

From the repository root:

```bash
python benchmark_6_solvers.py generated_problems \
  --config configs/relative_6_solver_benchmark.yaml \
  --output BenchmarkResults/relative_6_solver
```

The built-in settings match the supplied YAML, so `--config` is optional. SciPy + HiGHS
uses enlarged model-capacity gates for 1,000-variable QUBOs and disables presolve because
large linearizations can otherwise remain in an uninterruptible presolve phase. All
actual optimization parameters for Ocean, bifurcation, and transverse-route configurations
retain the solver adapter defaults.

Run a plan-only validation without importing or executing any solver:

```bash
python benchmark_6_solvers.py generated_problems --dry-run
```

## Progress and resume behavior

The terminal displays a task progress bar, successful/failed counts, current instance
and solver, and an estimated remaining time. `progress.json` is written atomically before
and after every solver run. Re-running the identical command resumes completed work;
`--retry-failures` explicitly retries previously failed tasks.

Use a different output directory when changing inputs or solver parameters. This avoids
mixing incompatible benchmark configurations.

`--snapshots N` enables solver-level incumbent snapshots, but it is separate from task
progress tracking and is disabled by default. In particular, bifurcation snapshots use
checkpoint replays and therefore change total work; leave them disabled for the cleanest
runtime comparison.

## Output structure

```text
BenchmarkResults/relative_6_solver/
  benchmark_config.json
  environment.json
  progress.json
  aggregate_report.{json,csv,md}
  all_solver_results.csv
  raw/<instance_id>/<solver_alias>.json
  instances/<instance_id>.{json,csv,md}
  failures/<instance_id>--<solver_alias>.json
```

Each instance report contains:

- energy, within-instance rank, best/tied flag, and wall time;
- absolute and relative energy gap to the best result for that instance;
- Hamming distance to the nearest best solution;
- full pairwise solution Hamming-distance and energy-difference matrices;
- explicit failure information when a solver does not complete.

The aggregate report states how many instances completed all six configurations. Its
per-solver summaries use each solver's successful runs, while pairwise outcomes use the
instances shared by the corresponding pair. It includes completion counts, wins/ties,
mean rank, mean energy, mean per-instance gap, mean relative gap, solution disagreement,
runtime statistics, and pairwise win/tie/loss counts. Raw solver JSON remains available
for detailed verification and contains the full binary sample.

## Benchmark interpretation

This is a native-default relative benchmark, not an equal-compute benchmark. The solver
defaults represent different workloads: HiGHS and bifurcation have time limits, while
Ocean uses fixed reads and sweeps. Compare both solution quality and wall time, and use
the within-instance gap/rank metrics as the primary cross-instance quality summary.
