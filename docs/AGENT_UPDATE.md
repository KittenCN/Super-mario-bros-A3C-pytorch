Agent update (2025-09-29)

Files created/modified by the debugging agent session:

- `docs/ENV_DEBUGGING_REPORT.md` - Detailed diagnostics and experiments relating to AsyncVectorEnv instability and nes_py compatibility issues.
- `src/envs/mario.py` - Added worker wrapper, nes_py runtime patches, per-worker diagnostics, file-lock serialization, explicit AsyncVectorEnv options.
- `train.py` - Added CLI flags (`--parent-prewarm-all`, `--worker-start-delay`, `--env-reset-timeout`, `--enable-tensorboard`), moved parent prewarm earlier, and adjusted multiprocessing start method logic.
- `README.md` - Added link to `docs/ENV_DEBUGGING_REPORT.md` and short note about Async vs Sync stability.

Notes:
- These changes were performed to improve robustness and observability of environment construction in systems where native libraries (nes_py, SDL) and NumPy/Python version mismatches can cause runtime failures.
- See `docs/ENV_DEBUGGING_REPORT.md` for the full experimental log and the recommended next steps.
