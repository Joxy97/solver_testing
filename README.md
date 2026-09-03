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

1. Edit `configs/random_qubo.yaml` to choose the problem parameters, count, and seed.
2. Edit `configs/solve_random_qubo.yaml` to choose solvers, parameters, and devices.
3. Run:

```bash
python generate.py configs/random_qubo.yaml
python solve.py configs/solve_random_qubo.yaml
```

Inputs go to `generated_problems/random_qubo/`; results go to
`BenchmarkResults/random_qubo/`. Both are ignored by Git, so old results are preserved.

To expose only physical GPUs 2 and 5 (renumbered inside the process to `cuda:0` and
`cuda:1`):

```bash
export CUDA_VISIBLE_DEVICES=2,5
python solve.py configs/solve_random_qubo.yaml
```

The standard runner is sequential. For concurrent multi-GPU runs, use one config and
output folder per process. See [the complete remote workflow](docs/RANDOM_QUBO_WORKFLOW.md)
for precise preparation, parameter selection, parallel execution, and result retrieval.

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
