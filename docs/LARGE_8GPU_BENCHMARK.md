# 100k-variable, eight-GPU benchmark

`benchmark_100k_8gpu.py` prepares one Random QUBO with 100,000 variables and quadratic
density `0.25`, then compares Simulated Bifurcation and Transverse-Route Geometric Flow
across eight CUDA devices. Every unspecified problem and solver setting retains its
adapter default.

Here `0.25` is the repository's `quadratic_density` parameter: each off-diagonal pair is
selected with probability 0.25. The remaining Random QUBO defaults are integer
coefficients in `[-10, 10]`, zero linear-coefficient probability `0.0`, and seed `0`.
Only the adapters' dense-allocation safety ceilings are raised to permit 100,000
variables; this does not change either algorithm.

## Why the problem is implicit

This problem has approximately 1,249,987,500 selected upper-triangular interactions.
Representing those terms as Python objects and JSON would require hundreds of gigabytes.
The runner instead writes a compact `problem_manifest.json` containing the distribution,
density, and seed. Each worker deterministically materializes the identical symmetric
matrix directly on its GPU in bounded temporary blocks.

The FP32 matrix occupies approximately 37.3 GiB on every device. The runner requires at
least 45 GiB free per GPU by default. This headroom covers generation blocks, trajectory
state, CUDA libraries, and the bifurcation optimizer. Use `--minimum-free-gib` to raise the
gate for the remote hardware. Reducing it can produce an out-of-memory failure.

## Complete remote command

From a fresh repository clone, with a CUDA-capable PyTorch wheel available for the host:

```bash
python -m pip install -r requirements.txt && python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 8; print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(8)])" && python benchmark_100k_8gpu.py --num-variables 100000 --quadratic-density 0.25 --devices 0,1,2,3,4,5,6,7 --expected-devices 8
```

By default the command uses `cuda:0` through `cuda:7` and writes:

```text
BenchmarkResults/random_qubo_100k_0p25_8gpu/
```

Use explicit device indices when necessary:

```bash
python benchmark_100k_8gpu.py --devices 1,2,3,4,5,6,7,8
```

The default `--agent-mode shard` distributes each solver's default total trajectory count
across eight devices: 128 bifurcation agents and 64 TRF agents in total. Deterministic
per-device random substreams prevent duplicate initial states. To run a complete default
agent batch on every GPU instead, use `--agent-mode replicate`; this performs eight times
as much trajectory work.

The workflow is resumable. Completed files under `device_runs/` are reused, so rerunning
the same command only retries missing or failed device/solver pairs.

`--block-rows` controls peak temporary allocation during deterministic generation. It is
part of the compact problem identity because changing the block partition changes the
seeded coefficient stream; use a new output folder when changing it or any other problem
or solver option. The runner rejects incompatible reuse automatically.

## Results and verification

The downloaded result folder is self-contained. It includes the compact problem manifest,
environment and benchmark configuration, per-device raw solutions, aggregated best
solutions, progress, failures, CSV summaries, JSON reports, and Markdown reports.

Final objective values are not trusted from FP32 dynamics. The runner regenerates the
integer QUBO coefficient stream and recomputes selected objectives with exact int64
accumulation. The aggregate wall time is the slowest device's solver time, while all
per-device times and energies remain available in `solver_metrics.device_runs`.

## Load downloaded results into the dashboard

Place the folder under `BenchmarkResults/`, then run from the repository root:

```bash
cd benchmark-dashboard
npm ci
npm run data -- ../BenchmarkResults/random_qubo_100k_0p25_8gpu
npm run dev
```

The dashboard data importer accepts this positional results path and also supports the
`BENCHMARK_RESULTS` environment variable.
