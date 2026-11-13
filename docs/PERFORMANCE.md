# Simulator throughput verification

For the subsequent affected-path fee refresh optimization, see
[FEE_CACHE_PERFORMANCE.md](FEE_CACHE_PERFORMANCE.md). The measurements below
describe the earlier full-state-copy and fee-arithmetic optimization.

Measured on 2026-11-12 against baseline commit
`5fc20c3ea823b073d50602094a94003fb6a020a1` using Python 3.12.11,
NumPy 2.4.6, Linux x86-64, and the project virtualenv. Runs were sequential;
the test suites did not run concurrently with the measurements. Host load is
not controlled, so these are local measurements, not a production throughput
guarantee. Machine-readable evidence is in `performance-results.json`.

## Unchanged workload and results

Both versions use the official scenario without overrides: **10,000 steps,
100 paths, 2,500 tick intervals, horizon 1.0**, and timestep 0.0001. The full
liquidity and fee grids remain in the simulation state. No market parameters,
actions, RNG draws, scoring rules, or sandbox restrictions were changed.

The periodic reference policy on public seed 42 produced exactly the same
100 per-path cumulative rewards and PnLs, action digest, and final-state digest.
Mean PnL was **71.57282579879903 token1** in both versions. The final-state
digest includes every state array, not only the public agent observation:

```text
d96202993fea8e87274d5f4898396f6399e2c31bd7b08cb39a3051421800fe95
```

A second full-size run on public seed 314 used asymmetric, varying ranges and
staggered deployment/rebalance schedules across the 100 paths. **All 10,001
state hashes matched exactly**, including the initial state and every step's
full state, action, reward, arrivals, termination flags, and diagnostics. All
per-path returns and PnLs also matched; mean PnL was
**127.30238416425254 token1** in both versions.

The second runs took 656.278 seconds before and 344.472 seconds after. These
are correctness traces with full-state hashing, a different policy, and many
partial rebalances; they are not directly comparable with the periodic-policy
profile below. Speedup depends on policy and measurement overhead.

## Profile

These figures cover the complete episode loop, including policy calls and
public observation copies, under cProfile. Environment construction and reset
are outside the measurement. This is one seed and one pass, without hosted
IPC, nsjail, queueing, or the second reproducibility pass.

| Measurement | Before | After |
|---|---:|---:|
| Episode wall time | 404.462 s | 40.926 s |
| Gross LP fee calculation, cumulative | 318.745 s | 15.590 s |
| Gross LP fee calls | 30,199 | 10,199 |
| Array copies directly in `env.step`, cumulative | 12.900 s | 0.012 s |
| Array copies directly in `env.step` | 220,000 | 10,000 |

The measured episode is **9.88 times faster**, with **89.88% less elapsed
time**. Fee computation originally accounted for 78.81% of elapsed time; the
full-state copies accounted for 3.19%. Cumulative profile rows are nested and
must not be added together.

## Implementation

- The exact built-in `PnL` reward snapshots only previous portfolio values.
  Other rewards, including subclasses of `PnL`, retain detached full-state
  snapshots.
- Fee division and multiplication gather only the LP range's intersection with
  the tracked grid. Grid-edge positions, zero liquidity, and empty ranges are
  handled without wrapping indices or counting a boundary interval twice.
- Weighted fees still occupy their original columns in a zero-filled full-width
  buffer before `np.sum`. Keeping this reduction shape preserves the original
  floating-point summation order. Compact-range summation would not provide that
  guarantee.
- The fees accrued after swaps are passed explicitly into the observation and
  valuation calculations. There is no persistent fee cache. Reset, standalone
  portfolio queries, and explicit observation refreshes still recompute fees.

The underlying full-grid fee arrays and the rebalance-time collection semantics
remain intact. This also supports nonuniform background liquidity without
assuming the nominal scenario's constant background amount.

## Reproduction

From this checkout, preserve the baseline in a separate directory:

```bash
mkdir -p /tmp/clmm-reference
git archive 5fc20c3ea823b073d50602094a94003fb6a020a1 | tar -x -C /tmp/clmm-reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_simulation.py \
  --root /tmp/clmm-reference --profile --output results/before-profile.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_simulation.py \
  --profile --compare results/before-profile.json --output results/after-profile.json
```

The script writes JSON and, with `--profile`, a sibling `.prof` file readable
with `pstats`. `--compare` fails unless the configuration, actions, every path's
returns/PnL, score, and complete final state match exactly.

For a second public seed and varying ranges/rebalance schedules, enable traces:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_simulation.py \
  --root /tmp/clmm-reference --seed 314 --policy mixed --trace --output results/before-mixed.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_simulation.py \
  --seed 314 --policy mixed --trace --compare results/before-mixed.json --output results/after-mixed.json
```

Traces hash every state array, action, reward, arrival indicator, termination
flag, and diagnostic after every step, plus the initial state. Hashing adds
overhead; use those runs for correctness rather than the headline timing.

## Regression checks and release use

The simulator suite passes **76 tests**, including upstream parity with
`SAIFE_UPSTREAM_PATH=../../SAiFE_gym`. The backend's CLMM, simulation-adapter,
and evaluation suites pass **36 tests**. New differential tests compare the
optimized fee calculation with the original dense formula using nonuniform
liquidity, widely varying fee magnitudes, grid edges, mixed holds/rebalances,
and reset. They also check standalone refreshes and custom reward snapshots.

A full-size backend check ran the public submission template on seed 42 with
paired reproducibility verification enabled. Both fresh agent processes produced
the original action digest and all 100 original per-path returns; the backend
reported `reproducibility_verified: true` and score `71.57282579879903`.
The complete two-pass call took **107.090 seconds** in development subprocess
mode, including the trusted driver and actual framed IPC. It did not use nsjail,
Redis, or the database and is not a production-host benchmark.

To repeat the backend check from `hackathon-backend/` with its own virtualenv:

```bash
SANDBOX_MODE=subprocess OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
import json
from pathlib import Path
from time import perf_counter
from worker import evaluator, simulation

root = simulation.resolve_path("../clmm-challenge")
reference = json.loads((root / "docs/performance-results.json").read_text())["profile_comparison"]["before"]
started = perf_counter()
result = evaluator.evaluate(
    "throughput-public-replay", (root / "examples/submission_template.py").read_text(),
    simulation_env_path=str(root), seeds=[42], verify_reproducibility=True,
)
elapsed = perf_counter() - started
assert result["status"] == "scored", result.get("message")
assert result["reproducibility_verified"] is True
assert result["score"] == reference["score"]
assert result["seed_results"][0]["returns"] == reference["returns"]
assert result["seed_results"][0]["action_digests"] == [reference["action_sha256"]] * 2
print({"elapsed_seconds": elapsed, "score": result["score"], "verified": True})
PY
```

This changes simulator source fingerprints even though the checked results are
identical. Existing registered competitions must retain their original worker
image. For a new release, rebuild the worker and participant download from the
same reviewed source, then register the new evaluation suites. No deployment,
official seed registration, or database changes are part of this benchmark.
