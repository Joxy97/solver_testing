# 100k density sweep checkpoint

This checkpoint was captured after the supervised run was stopped cleanly.
It contains all 48 calibration runs, the selected parameter file, and two
completed production instances (`0p01` seeds 0 and 1). Seed 2 was interrupted
before its first device result completed. The benchmark runner detects completed
per-device JSON files and only runs missing work.

## Restore on a new Vast.ai instance

Clone the repository into `/workspace/solver_testing`, copy the checkpoint
archive to the instance, then run:

```bash
cd /workspace/solver_testing
tar -xzf /path/to/qubo_density_resume_20260903.tar.gz -C .
source /venv/main/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
install -m 755 resume/qubo-density-sweep.sh /opt/supervisor-scripts/qubo-density-sweep.sh
install -m 644 resume/qubo_density_sweep.conf /etc/supervisor/conf.d/qubo_density_sweep.conf
supervisorctl reread
supervisorctl update
supervisorctl start qubo_density_sweep
supervisorctl status qubo_density_sweep
```

Follow progress with:

```bash
tail -f /workspace/solver_testing/BenchmarkResults/density_sweep_runs/supervisor.log
```

The finished dashboard-importable suite will be written to
`BenchmarkResults/density_sweep_100k` after all 60 instances complete.
