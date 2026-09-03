# Random QUBO experiment on a multi-GPU remote machine

## 1. Clone and prepare Python

```bash
git clone https://github.com/Joxy97/solver_testing.git
cd solver_testing
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer. After a new SSH login, reactivate with
`source .venv/bin/activate`.

## 2. Check GPU access

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

If CUDA is false, install the CUDA-enabled PyTorch build matching the machine using the
official PyTorch installation selector. The NVIDIA driver must work.

## 3. Choose Random QUBO parameters

Edit `configs/random_qubo.yaml`:

- `count`: number of independently seeded instances.
- `base_seed`: seed of the first instance; later instances use consecutive seeds.
- `num_variables`: number of binary variables.
- `quadratic_density`: probability of each variable pair being present (0 to 1).
- `distribution`: `integer`, `uniform`, or `normal`.
- `coefficient_min` and `coefficient_max`: integer/uniform bounds.
- `normal_scale`: standard deviation for a normal distribution.
- `zero_linear_probability`: probability that a linear coefficient is omitted.

Run `python generate.py --describe random_qubo` to see the complete parameter reference.
A dense QUBO has `n(n-1)/2` possible quadratic terms, so test a small configuration first.

## 4. Choose solvers and GPUs

Edit `configs/solve_random_qubo.yaml`. Remove unwanted solver entries and change their
parameters. Inspect all choices with:

```bash
python solve.py --list
python solve.py --describe simulated_bifurcation
python solve.py --describe transverse_route
python solve.py --describe ocean_sa
```

GPU solvers accept `device: cuda:0`, `cuda:1`, and so on. You can either put physical
indices directly in YAML or restrict the GPUs visible to the process, which is safer on a
shared machine:

```bash
export CUDA_VISIBLE_DEVICES=2,5
```

Physical GPU 2 then becomes `cuda:0`, and physical GPU 5 becomes `cuda:1`. Re-run the
Python GPU check above to confirm. On a managed cluster, use only scheduler-assigned GPUs;
Slurm commonly sets `CUDA_VISIBLE_DEVICES` for you.

## 5. Generate and smoke-test

```bash
python generate.py configs/random_qubo.yaml
ls generated_problems/random_qubo
python solve.py --problem generated_problems/random_qubo/INSTANCE.json --solver transverse_route --set device=cuda:0 --set max_steps=10 --output BenchmarkResults/smoke_test
```

Replace `INSTANCE.json` with one generated filename. After it succeeds, run the experiment:

```bash
python solve.py configs/solve_random_qubo.yaml
```

With `continue_on_error: true`, a failed solver is reported and later runs continue.

## 6. Run multiple GPUs concurrently (optional)

The standard config runs entries sequentially; choosing different devices controls
placement, not concurrency. To run concurrently, copy the solver config once per GPU.
Give each copy one GPU solver entry and a unique `output_folder`, such as
`../BenchmarkResults/random_qubo/gpu0` and `../BenchmarkResults/random_qubo/gpu1`.

```bash
CUDA_VISIBLE_DEVICES=2 python solve.py configs/solve_gpu0.yaml > gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python solve.py configs/solve_gpu1.yaml > gpu1.log 2>&1 &
wait
```

Use `device: cuda:0` in both configs because each process sees one GPU. This can run
different solvers or parameter sets on the same instances. To divide instances, use
non-overlapping `problem_files` globs or shard the files into separate directories. Never
have concurrent processes write to the same output folder. Use your scheduler, `tmux`, or
`screen` so an SSH disconnect does not terminate a long run.

## 7. Inspect and retrieve results

Every result JSON records the instance, solver parameters, status, binary sample, verified
energy, timing, device, and solver metrics. Generated inputs include `manifest.csv`.

Run these retrieval commands on your local machine:

```bash
rsync -av --progress user@remote-host:/path/to/solver_testing/BenchmarkResults/random_qubo/ ./BenchmarkResults/random_qubo/
rsync -av --progress user@remote-host:/path/to/solver_testing/generated_problems/random_qubo/ ./generated_problems/random_qubo/
```

Also retain the edited YAML files. Together, the configuration, seeds, generated inputs,
and result JSON files provide a complete experiment record.
