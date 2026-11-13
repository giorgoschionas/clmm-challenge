# Fee refresh optimization

This change reuses gross LP fee totals for paths whose relevant inputs are
unchanged. The comparison is exact, with no tolerance. It checks detached copies
of LP liquidity, range bounds, and the pool liquidity and token0/token1 fees
within that range. Official ranges span at most 100 of the 2,500 tick intervals.

Checking inputs, rather than relying exclusively on swap events, also detects
in-place edits by local callers and custom models. Invalid padding outside the
position is excluded. A change in compact-array shape invalidates the batch
cache conservatively. Entry snapshots are subtracted afresh every time.

Only changed paths undergo division, weighting and the original 2,500-column
fee reduction. Keeping that width and arithmetic order preserves floating-point
results. Non-contiguous or non-float64 liquidity arrays use the uncached path.
Reset clears the cache. Explicit fee and portfolio queries always recompute;
they never trust cached totals. The cache is internal, outside the public state.
Portfolio valuation, per-step rewards, swap mechanics and RNG draws are unchanged.

The repeated measurements below are the preferred timing comparison for this
change. With other project work left idle, median unprofiled standalone time
fell **21.6%** and median production-nsjail evaluation time, including replay,
fell **20.6%**. Every result matched exactly.

## Workload and scope

Measured on 2026-11-13 against commit
`f7ecf0ad96ecedcc9fd2f336b331b87db91385f8`, the already optimized simulator.
Every full-size run uses 10,000 steps, 100 paths, 2,500 tick intervals and horizon
1.0 without configuration overrides. Public seeds only: 42 for the periodic
reference strategy and 314 for mixed ranges and staggered rebalances.

The standalone profile covers one episode with cProfile enabled. The mixed
correctness traces additionally hash all state arrays, actions, rewards, arrivals,
termination flags and diagnostics at every step. Trace timings include that
substantial verification overhead and are not throughput measurements.

Production-sandbox benchmarks use the backend's real `evaluator.evaluate` with
`verify_reproducibility=True`, hence two fresh nsjail agents per evaluation.
They include initialization, simulation, framed IPC, scoring, teardown and replay
comparison. They exclude Docker/Python startup, HTTP, queueing and database work.
The worker is limited to 2 CPUs, 2 GiB and 128 PIDs, with the production namespace
capabilities, seccomp policy, agent limits and network isolation. The reference
and optimized images share the backend, Python, dependencies and nsjail binary.
All measurements run locally under WSL2; these are not Hetzner throughput
guarantees. The initial measurements overlapped other project work. The repeated
measurements below were requested with that work left idle.

## Repeated measurements with other project work idle

Repeated on 2026-11-13 using the same simulator source fingerprints and existing
production worker images as the initial comparison. The laptop exposes an Intel
Core i5-10210U with 4 cores / 8 logical CPUs to WSL2. No simulator, policy,
scenario or sandbox settings changed for this repeat.

Each mode had one full warm-up per version, excluded from the summary in advance,
then **five measured runs per version**. Measured pairs alternated order:
before/after, after/before, before/after, after/before, before/after. All runs
were sequential, with no concurrent tests or image builds. All measured samples,
including the slow first baseline sandbox evaluation, are retained. Neither
cProfile nor full-step state tracing was enabled in these timing runs.

Every episode uses the periodic reference strategy and public seed **42**, with
**10,000 steps, 100 paths, 2,500 ticks and horizon 1.0**. Each sandbox evaluation
includes two fresh nsjail agents: the scored episode and its reproducibility
replay. These timings cover one seed's batch, rather than the full three-seed
live suite.

| Measurement | Before median (min–max) | After median (min–max) | Median time reduction |
|---|---:|---:|---:|
| Standalone episode, one pass | 24.852 s (22.690–27.040) | 19.475 s (18.068–22.198) | 21.6% |
| Production nsjail evaluation, including replay | 76.021 s (72.169–102.214) | 60.334 s (58.916–63.807) | 20.6% |

The ratios of before/after medians are **1.28x** for standalone and **1.26x**
for nsjail. The optimized version was faster in all five corresponding pairs
in both modes. Median reductions calculated within each pair are 17.5% and
18.5%, respectively; these differ from the ratio-of-medians calculation above.
These samples support an improvement for this workload, while still showing
variation in laptop timing.

| Pair | Standalone before | Standalone after | Nsjail before, with replay | Nsjail after, with replay |
|---|---:|---:|---:|---:|
| 1 | 23.199 s | 18.068 s | 102.214 s | 63.403 s |
| 2 | 25.076 s | 22.198 s | 73.129 s | 59.606 s |
| 3 | 27.040 s | 19.475 s | 72.169 s | 58.916 s |
| 4 | 24.852 s | 20.552 s | 78.064 s | 63.807 s |
| 5 | 22.690 s | 18.724 s | 76.021 s | 60.334 s |

All **10 measured standalone episodes** and **20 measured sandbox passes**
matched the original reference exactly: mean PnL, all 100 per-path returns,
actions and full final state. Standalone terminal PnLs and backend per-path PnL
percentages also matched. All **10 measured sandbox evaluations** passed the
backend's reproducibility check. The two standalone warm-ups and four sandbox
warm-up passes also passed. Mean PnL remained **71.57282579879903 token1**.

The sandbox checks also verified identical Python/dependency versions, backend
source, agent source, nsjail binary, harness and resource limits across versions.
Container CPU accounting recorded **zero quota throttling** in all 12 sandbox
evaluations, including warm-ups. The 2-CPU quota does not reserve exclusive cores;
CPU frequency and Windows background activity were not controlled or measured.
Host WSL load and CPU-pressure snapshots are retained with each run.

Standalone timing covers the episode loop, including policy, observations,
action hashing and reward accumulation; initialization and final-state hashing
are outside that timer. Sandbox timing covers the complete backend evaluator
call, including both passes, initialization, IPC, scoring, teardown and replay
comparison, plus the harness's final-state digest checks. It excludes Docker/
Python startup, HTTP, queueing and database work. Compare before versus after
within each mode; the two modes have different timing scopes and Python runtimes.

Full reports for every measured run and warm-up, exact commands, image/source
fingerprints, host snapshots, CPU accounting and summary statistics are retained
in [fee-cache-idle-benchmark-results.json](fee-cache-idle-benchmark-results.json).
Local orchestration scripts and logs are under `results/fee-cache-idle-20261113/`
(ignored by Git). The earlier separate mixed-strategy full-step trace and test
results remain documented below; they were not rerun as part of these timings.

## Initial standalone profile

Periodic reference strategy, public seed 42, one profiled episode per version:

| Measurement | Before | After | Reduction |
|---|---:|---:|---:|
| Episode, including policy and observations | 26.881 s | 24.494 s | 8.9% |
| Fee accrual, including cache checks | 10.193 s | 6.600 s | 35.3% |
| Gross fee computation only | 10.195 s | 2.332 s | 77.1% |

Gross computation is nested within fee accrual; these rows must not be added.
The end-to-end simulator speedup in this profiled sample is **1.10x**.
All 100 returns and terminal PnLs, action digest, score and full final-state
digest match exactly. The score is **71.57282579879903 token1**.

## Exactness and regression checks

The periodic run matches all 100 per-path returns and terminal PnLs, the action
digest and every final state array exactly. Its final-state SHA-256 is
`d96202993fea8e87274d5f4898396f6399e2c31bd7b08cb39a3051421800fe95`.

The mixed strategy on seed 314 matches all **10,001 state hashes**, including
reset and every step's state, actions, rewards, arrivals and diagnostics. Its
score is **127.30238416425254 token1**. SHA-256 of the concatenated state hashes:
`557cf7910bd97e2b35aa92f98a4c351772c2f4bb8b75b7f6625e0ac8c3276b53`.

All **86 simulator tests** pass, including upstream parity, and all **36 backend
CLMM/adapter/evaluation tests** pass. The new differential checks cover selective
refresh, out-of-range fees, external mutations to fees/liquidity/positions,
entry snapshots, lifetime fee deltas, unusual array layouts, explicit queries
and reset. The new worker image also passes the production nsjail isolation
smoke test: forbidden process/network syscalls, inaccessible secrets/procfs,
and isolated read-only mounts.

## Initial production nsjail results

Two evaluations per version, each including both fresh-agent passes on public
seed 42. Runs were sequential in this order: before 1, after 1, after 2, before 2.
All eight passes matched the reference action digest, all 100 returns and full
final-state digest. All four evaluations passed the backend reproducibility check
and exactly matched before/after score and per-path PnL percentages.

| Version / repeat | First pass | Replay pass | Complete evaluation |
|---|---:|---:|---:|
| before 1 | 116.491 s | 35.898 s | 152.630 s |
| after 1 | 35.907 s | 29.459 s | 65.516 s |
| after 2 | 41.279 s | 41.383 s | 82.807 s |
| before 2 | 38.774 s | 36.201 s | 75.161 s |

In this initial set, optimized evaluations took **65.516 and 82.807 seconds**, versus
**152.630 and 75.161 seconds** before. The first baseline pass was unusually slow;
with this variability and only two samples per version, these measurements do
**not establish a stable end-to-end speedup**. The isolated profiled simulator
shows the smaller improvement reported above. No sandbox ratio from this initial
set is used as a headline performance claim; the repeated comparison above
provides the subsequent measurements with other project work left idle.

The optimized workers peaked at approximately **98.8 MiB** of container memory.
The evaluator still uses the production 512 MiB agent address-space limit,
60-second agent CPU limit, 5-second per-action timeout and 1,800-second per-pass
wall timeout. Python, installed packages, backend code, nsjail binary and resource
limits were verified identical across all four measurements.

Machine-readable results, per-path outcomes, source fingerprints, image IDs and
Docker commands are in [fee-cache-performance-results.json](fee-cache-performance-results.json).
The detailed local logs and complete traces remain under
`results/fee-optimization-20261113/` (ignored by Git).

## Reproduce

From `clmm-challenge`, archive the baseline and run sequentially:

```bash
mkdir -p /tmp/clmm-fee-reference
git archive f7ecf0ad96ecedcc9fd2f336b331b87db91385f8 | tar -x -C /tmp/clmm-fee-reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_simulation.py \
  --root /tmp/clmm-fee-reference --profile --output results/fee-before.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_simulation.py \
  --profile --compare results/fee-before.json --output results/fee-after.json
```

Repeat with `--seed 314 --policy mixed --trace` and distinct output files to
compare all 10,001 full-state hashes, including reset. Run regression checks with:

```bash
SAIFE_UPSTREAM_PATH=../../SAiFE_gym .venv/bin/python -m pytest
cd ../hackathon-backend
.venv/bin/python -m unittest tests.test_clmm tests.test_simulation_parity tests.test_evaluations
```

`scripts/benchmark_nsjail.py` runs inside a production worker image. It requires
`--reference` (the baseline standalone periodic/seed-42 JSON), `--source-sha256`
(the source digest recorded by the corresponding standalone profile), and
`--output`. Supply `--compare` with an earlier sandbox report to require exact
before/after outcomes and identical backend/runtime/nsjail provenance. Mount the
harness and reports into the worker; source hashes prevent accidentally timing
an older image. Exact Docker commands and image IDs are retained in the local
`results/fee-optimization-20261113/` metadata files.

For the repeated throughput comparison, omit `--profile` and `--trace` from the
standalone commands. Use one complete warm-up per version and mode, then five
measured pairs in the order documented above, always with distinct output files.
Keep `--compare` enabled for every standalone run and
`verify_reproducibility=True` in the sandbox harness. The exact commands and
run schedule used here are in `fee-cache-idle-benchmark-results.json`.

This work does not deploy to Hetzner or alter registered evaluation suites.
The simulator source fingerprint changes, as with the previous optimization.
