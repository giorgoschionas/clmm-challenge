# CLMM Challenge

A standalone liquidity-provision competition extracted from SAiFE_gym. Only
NumPy and Gymnasium are required at runtime. There is no dependency on a sibling
SAiFE_gym checkout, PyTorch, Stable-Baselines3, pandas, or training wrappers.

## Getting started

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp submission_template.py my_agent.py
.venv/bin/python -m clmm_challenge.evaluate --agent my_agent.py
.venv/bin/python -m pytest
```

Practice runs 100 paths for each public seed (42, 314, 2718). Use
`--submissions-dir submissions` to compare files, `--seeds 42` for a shorter run,
and `--no-baselines` to omit the reference policies. CSV output defaults to
`results/leaderboard.csv`. Local evaluation executes trusted source files in
the grading process; it is not an untrusted-code sandbox. Hosted evaluation
uses the backend's separate agent process and nsjail in production.

## Agent interface

Submit one Python file, at most 64 KiB, defining `Agent.__init__(self, config)`
and `Agent.get_action(self, state)`. The starter uses only NumPy. Python standard
library imports allowed by the hosted sandbox are also available; training
libraries and external data files are not part of the challenge runtime.
For compatibility, a differently named `...Agent` must subclass the small
`from concentrator import Agent` interface supplied by the evaluator.

`config` is a detached namespace of public scalar scenario parameters. It never
contains the live environment, seed, or RNG. Return a finite numeric array of
shape `(num_trajectories, 3)`: `[lower_offset, upper_offset, hold_flag]`.
The first two entries are tick offsets from the current pool tick, rounded and
clipped to `[-tau, tau-1]` and `[-tau+1, tau]`; invalid ordering is corrected by
the engine. A positive hold flag keeps the existing position. A nonpositive
flag deploys/rebalances. Decisions occur every step; there is no stride wrapper.

The observation contains copied arrays of shape `(num_trajectories,)`:

| Field | Meaning |
|---|---|
| `sqrt_price` | Square root of pool price; square it to obtain price |
| `current_tick` | Current absolute pool tick |
| `midprice` | External reference price |
| `time` | Elapsed simulation time |
| `lp_tick_lower`, `lp_tick_upper` | Absolute LP range bounds |
| `lp_ever_deployed` | Whether liquidity has previously been deployed |
| `gas_cost` | Rebalance cost |
| `portfolio_value` | Wealth marked at the external reference price |
| `active_liquidity` | Liquidity at the current pool interval |
| `lp_token0_amount` | Token0 inventory in the LP position, used in scoring |
| `lp_alpha` | LP token0 value fraction |

## Fixed v1 market and scoring

The factory `clmm_challenge.challenge.create_environment` is shared by local
practice and the backend. `ScenarioConfig` is the authoritative configuration.

| Parameter | Value |
|---|---|
| Paths / steps / horizon | 100 / 1,000 / 1.0 |
| Initial price / wealth | 100 / 1,000 token1 |
| Midprice | Geometric Brownian model; drift 0, volatility 0.03 |
| Arrival coefficients per side | Floor 10, baseline 300, liquidity 0, arbitrage 4,000 |
| Liquidity kernel | Beta 0.5, window 10, normalization 1,000,000 |
| Price impact | Liquidity-depth; fixed gross trade notional 250 token1 |
| Depth window / minimum depth | 10 / 1e-12 |
| Ticks / spacing / background liquidity | 7,000 / 1.0001 / 100,000 per tick |
| LP offset limit | ±50 ticks |
| Pool fee | 0.003 |
| Gas / additional rebalance swap fee | 2 / 0 |
| Inventory penalty | Coefficient 0.4, exponent 2, terminal penalty 0 |

Preserves upstream's Euler midprice update, exact Poisson probability
`1-exp(-intensity*dt)` with at most one arrival per side per step, directional
liquidity-depth impact with stochastic tick rounding, and fee allocation.
The first deployment is free; subsequent rebalances pay gas. Never deploying
keeps cash and has zero reward.

Each step's reward is `V_next - V_current - 0.4 * dt * token0_inventory_next**2`.
The **score** is the mean cumulative reward over all paths and seeds; higher
is better. **Raw PnL** is final wealth minus initial wealth and is displayed
separately. Fees and costs already enter wealth; do not deduct them twice.
The best valid submission per participant is ranked on the hosted leaderboard.

## Backend connection

The backend routes `challengeSlug: "clmm"` to engine `clmm-v1`, imports this
package via `CLMM_ENV_PATH=../clmm-challenge`, and runs the agent in its existing
sandbox. Hosted evaluation requires `CLMM_SEEDS` in the process environment;
it never falls back to public practice seeds. CLMM is seeded upcoming with no
dates. Open it deliberately using the backend's CLMM-specific status/schedule
settings. Existing Concentrator submissions continue using their old engine.

## Maintaining the extraction

`UPSTREAM.json` records upstream commit and source-file hashes. The engine is
extracted from commit `138c5b39a2e7066df9e6e79b297230221f597fb6`.
Imports were renamed; alternative concrete models, taker-order execution,
unused helpers, default model construction, and training integrations were
removed. The retained numerical algorithms were not rewritten. Required bases
and RNG/reset helpers remain, even when their interfaces support more than
this single scenario. `simulation_core.py` retains shared state transitions.

Run optional upstream parity tests with
`SAIFE_UPSTREAM_PATH=/path/to/SAiFE_gym .venv/bin/python -m pytest`.
They compare complete states, rewards, arrivals, and termination for identical
seeds and actions. An upstream checkout is only a development oracle, never a
runtime dependency. Import new upstream changes deliberately and rerun parity;
do not silently change an active competition's rules.

Original contributions retain MIT terms; inherited mbt_gym portions retain
BSD-3-Clause terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
