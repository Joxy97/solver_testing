# QUBO Experiment Toolkit

This project generates reproducible QUBO instances and solves them with CPU and GPU
solvers. Experiments use editable YAML files, so no Python changes are needed.

## Start here

Python 3.10 or newer is recommended. From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, verify `nvidia-smi` works. Then confirm PyTorch sees CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If it does not, install the appropriate CUDA-enabled PyTorch wheel from the official
PyTorch installation selector.

## Run a Random QUBO experiment

1. Edit `configs/generate/random_qubo.yaml` to choose the problem parameters, batch, count, and seed.
2. Edit `configs/solve/solve_random_qubo.yaml` to choose named solver instances, parameters, and devices.
3. Run:

```bash
python generate.py configs/generate/random_qubo.yaml
python solve.py configs/solve/solve_random_qubo.yaml
```

Each experiment is self-contained: inputs go to `generated_problems/BATCH/qubos/`, and
solving creates the sibling `generated_problems/BATCH/results/` with one named solution
JSON per problem-and-solver pair.
The default batch format is `random_qubo_{variables}_{density}`. Existing runs are never
overwritten: a colliding batch becomes `BATCH(2)`, then `BATCH(3)`, and so on. Repeated
solves use `results(2)`, `results(3)`, and so on. Generated instance filenames use their
seed, such as `seed_42.json`.

Parallel execution is controlled directly in the solve YAML. A solver's `devices` list is
a GPU pool: different QUBOs are sent concurrently to free devices, with at most one task
per GPU. `replicas` enables independent multi-start runs of each QUBO.

```yaml
execution:
  mode: parallel
  order: problem_first
  cpu_workers: 2

solvers:
  - type: transverse_route
    instance_name: trf_pool
    devices: [cuda:0, cuda:1, cuda:2, cuda:3]
    replicas: 1
    parameters:
      agents: 64
      max_steps: 1000
```

To expose only physical GPUs 2 and 5 (renumbered inside the process to `cuda:0` and
`cuda:1`):

```bash
export CUDA_VISIBLE_DEVICES=2,5
python solve.py configs/solve/solve_random_qubo.yaml
```

See [the complete remote workflow](docs/RANDOM_QUBO_WORKFLOW.md) for preparation, queue
ordering, multi-GPU execution, replicas, and result retrieval.

## Examine a batch in the dashboard

The dashboard reads one local batch at a time. It includes energy and runtime comparisons,
a solver leaderboard, problem structure and coefficient charts, searchable result tables,
and individual solution details. Files stay in the browser session.

```bash
cd dashboard
npm install
npm run dev
```

Open the displayed local URL, choose `Load batch folder`, and select a folder such as
`generated_problems/random_qubo_1000_0.01`. The folder must contain `qubos/` and may
contain `results/`, `results(2)/`, and later result sets.

## Discover available options

```bash
python generate.py --list
python generate.py --describe random_qubo
python solve.py --list
python solve.py --describe simulated_bifurcation
```

Interactive guided modes are also available with `python generate.py` and
`python solve.py`.

Reference documentation:

- [Problem generators](docs/PROBLEMS.md)
- [Solver parameters and limitations](docs/SOLVERS.md)
- [Validation](docs/VALIDATION.md)
- [Adding a solver](docs/new_solver.md)

## Project layout

```text
configs/                 editable experiment settings
docs/                    detailed user documentation
problems/                QUBO generators and validation
solvers/                 solver adapters
generate.py              generate reproducible problem JSON files
solve.py                 solve problem files and save result JSON files
generated_problems/      local generated inputs (not committed)
BenchmarkResults/        local experiment results (not committed)
```
