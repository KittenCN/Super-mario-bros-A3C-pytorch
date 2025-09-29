# Environment & AsyncVectorEnv Debugging Report

Date: 2025-09-29

This document captures the diagnosis, experiments, observations and suggested fixes performed while investigating instability when constructing `AsyncVectorEnv` for the Super Mario Bros A3C training pipeline. It is intended to survive repository reloads and provide a single source-of-truth for the current state.

## Summary of the problem

- Symptom: Training sometimes fails to construct an `AsyncVectorEnv` in the parent process. The parent prints:
  - "Async vector env construction timed out after <N>s; retrying with synchronous vector env." and then falls back to synchronous env.
- Additional symptom: In single-process reproduction, `nes_py` raised OverflowError: `Python integer 1024 out of bounds for uint8` on Python 3.12 + NumPy 2.x.
- Observation: diagnostic logs written by worker thunks show workers reach `calling_mario_make` and then either crash with OverflowError (if patches not applied) or become silent (hang) after `calling_mario_make` (if patches applied). This indicates blocking inside `mario_make` / nes_py / native initialization.

## Where diagnostics were added

- `src/envs/mario.py`:
  - Per-worker diagnostic files under `/tmp/mario_env_diag_<parentpid>_<ts>/worker_idx<pid>.log` capturing stage timestamps and events: `start`, `patch_start`, `patch_uint8_ok`, `patch_ramdtype_ok`, `import_start_gsm`, `import_done_gsm`, `calling_mario_make`, and additional worker-side logs.
  - Runtime patches: `_patch_legacy_nes_py_uint8()` and `_patch_nes_py_ram_dtype()` to mitigate numpy uint8 → Python int overflow.
  - A wrapped worker (`_wrap_worker`) that constructs the env inside the worker, logs progress and delegates to the original gym/gymnasium worker loop if available. The wrapper also implements a filesystem lock (`mario_make.lock`) to serialize heavy initialization.
  - `create_vector_env` tries to pass `worker=_wrap_worker` when constructing `AsyncVectorEnv` and sets `shared_memory=False` and `context` to the parent start method.

- `train.py`:
  - CLI flags added: `--parent-prewarm`, `--parent-prewarm-all`, `--worker-start-delay`, `--env-reset-timeout`, and `--enable-tensorboard` (default off).
  - Parent prewarm was moved to run before vector env construction to warm native libs in the parent.

## Observed behaviour (chronology)

1. On an unpatched codepath (no nes_py patches), single-process reproduction of `gym_super_mario_bros.make(...)` raised `OverflowError` pointing into `nes_py/_rom.py` (numeric dtype issue).
2. After adding nes_py runtime patches (in worker thumbs and parent prewarm), the OverflowError disappeared in many reproductions, but parent still reported Async construct timeouts in multiple runs.
3. Diagnostic files show workers successfully reach `calling_mario_make` and sometimes `lock_acquired`; many runs stop producing further logs after `calling_mario_make` -> indicating the worker is blocked inside `mario_make()` or subsequent native initialization.
4. Attempts tried and their immediate effects:
   - Changing multiprocessing start method ordering (tried spawn, forkserver, then fork as primary) — mixed results: `fork` helps prewarm inheritance but not fully reliable; `spawn` can avoid some deadlocks but suffers from repeated heavy initializations.
   - `parent-prewarm-all` (sequentially build 1-env instances in parent) — reduced some races but did not fully eliminate Async construct timeouts.
   - `worker_start_delay` (staggering worker startup) — reduces probability of thundering herd but not a guarantee.
   - Passing `shared_memory=False` to `AsyncVectorEnv` — avoids shared memory races/EBADF but does not remove blocking in `mario_make`.
   - Implemented `wrap_worker` to expose worker-side errors/tracebacks into diagnostics — increased visibility but many runs still blocked inside `mario_make`.
   - Added a file lock around `mario_make` in the worker to serialize native init — this reduced concurrent initialization pressure and allowed some workers to complete construct, but parent sometimes still timed out because some workers take long.

## Experimental results (what was tried & outcome)

- nes_py patches (uint8 → int key properties, and RAM dtype astype to int64):
  - Implemented and executed in worker thunk and parent prewarm.
  - Result: OverflowError reproducible before patch; after patch the error largely disappears in single-process tests.

- Parent prewarm (single + all):
  - Moved to run before AsyncVectorEnv construction.
  - `--parent-prewarm` (single env) and `--parent-prewarm-all` (sequentially prewarm every worker config) both implemented.
  - Result: Helps but does not fully eliminate Async construction timeouts in all runs.

- Multiprocessing start method experiments:
  - Tried `spawn`, `forkserver`, `fork` ordering.
  - Result: `fork` allows prewarm inheritance but can be unsafe if threads are present; `spawn` avoids fork issues but multiplies first-time init cost.

- Wrapped worker + file lock:
  - Implemented `_wrap_worker` to log worker events; added `mario_make.lock` to serialize heavy init.
  - Result: Increased visibility (per-worker logs have worker_construct_start/lock_acquired). Lock reduces concurrency but some workers still take long enough to cause parent timeout. Parent still falls back to Sync in some runs.

- Passing `worker` to AsyncVectorEnv and `shared_memory=False`:
  - Implemented to reduce FD/shm races.
  - Result: Avoids EBADF in teardown; doesn't eliminate long blocking in `mario_make`.

## Current diagnosis

- The core blocker appears to be heavy native initialization in `mario_make()` / nes_py (SDL, ROM parsing, C extensions) which, when performed concurrently across multiple processes, can block/serialize internally or contend on system resources (I/O, locks, SDL display resources) causing some workers to take long time or hang.
- While nes_py numeric dtype bugs were real and patched, they are not the sole cause of the Async timeout behaviour.
- Serializing init (lock) or staggering helps but doesn't fully guarantee parent-timeout-free behaviour: some environments still take longer than the parent's configured timeout (60s by default), so parent falls back to Sync.

## Recommendations & possible solutions (with status)

1. Immediate / conservative (Recommended for 10-hour runs):
   - Force synchronous vector env (`--force-sync`) for long runs. This is stable and has been verified in short tests. Trade-off: less sampling throughput but high reliability.
   - Use `--parent-prewarm-all` to warm libraries in parent before training. This reduces rare first-use stalls.
   - Increase `--env-reset-timeout` to a generous value (e.g., 300s) to reduce unintended fallbacks during early init.
   - Status: Implemented and tested (Sync works reliably under short runs).

2. Medium-term (improve Async reliability without sacrificing throughput):
   - Implement sequential worker startup (strict ordered start): start worker 0, wait for worker-ready signal (explicit IPC), then start worker 1, etc. This is more invasive but avoids thundering herd and is likely to stabilize Async. I can implement this as the next step.
   - Alternatively, implement batched startup (start 2–4 workers at a time) if fully sequential is too slow.
   - Status: partially implemented ideas (we have wrap_worker and lock), but full sequential-start orchestration not yet implemented.

3. Deeper / last-resort solutions:
   - Investigate upgrading/downgrading native dependencies (nes_py, gymnasium/gym, gymnasium-super-mario-bros) to versions known to be stable on Python 3.12 + NumPy 2.x, or pin to Python 3.10/3.11 runtime where problematic behaviours are not present.
   - If possible, move heavy native init to parent and serialize or snapshot state that workers can reuse (this requires intimate knowledge of nes_py internals and may not be feasible).
   - Status: Not implemented (requires external package changes or deeper reverse-engineering of native libs).

4. Observability & safety improvements (small changes that help long runs):
   - Periodic checkpoints (`--checkpoint-interval`) and monitor process (`--monitor-interval`) to capture resource metrics.
   - Persistent logging of per-worker diagnostics (already implemented under `/tmp/mario_env_diag_*`).
   - Status: Implemented.

## How to proceed now (concrete options)

- If you want to run a 10-hour job right away and prefer safety: run with `--force-sync`, `--parent-prewarm-all`, and an expanded `--env-reset-timeout` as suggested above.
- If you want to use Async for throughput, I recommend I implement sequential worker startup (option 2 above). I can do that next and run a short 5–15 minute smoke test to validate stability.

## Locations of artefacts & logs

- Code changes made:
  - `src/envs/mario.py`: worker wrap, nes_py patches, diag logging, serialization lock, AsyncVectorEnv options.
  - `train.py`: CLI flags, parent prewarm ordering, start-method adjustments.
  - `src/utils/logger.py`: TensorBoard default-off changes.

- Diagnostics:
  - Per-worker logs: `/tmp/mario_env_diag_<parentpid>_<ts>/*` (contains `env_init_*.log` and `worker_idx*.log`).

## Immediate next steps I can implement (pick one)

1. Implement sequential worker startup (strongly recommended if you need Async) — I'll implement and run a short test.
2. Increase default timeouts, worker_start_delay, lock timeout and run another battery of tests (quick).
3. Stop here and use synchronous mode for long runs.

---
Generated and updated by the in-repo debugging session on 2025-09-29. Keep this file under `docs/` so it's available after session restarts.
