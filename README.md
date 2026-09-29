# CLMM Challenge

Build a Python strategy that provides concentrated liquidity, earns trading
fees, and manages inventory risk and rebalancing costs. Evaluate it locally
against public practice seeds and compare it with reference policies.

## Getting started

Requires Python 3.11 or newer. Installation includes NumPy and Gymnasium.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp submission_template.py my_agent.py
.venv/bin/python -m clmm_challenge.evaluate --agent my_agent.py
```

Practice runs 100 paths for each public seed (42, 314, 2718). Use
`--submissions-dir submissions` to compare files, `--seeds 42` for a shorter run,
and `--no-baselines` to omit the reference policies. CSV output defaults to
`results/leaderboard.csv`. Local evaluation executes strategy files on your
machine, so run only code you trust. Hosted submissions run in an isolated
sandbox.

## Agent interface

Submit one Python file, at most 64 KiB, defining `Agent.__init__(self, config)`
and `Agent.get_action(self, state)`. The starter uses only NumPy. Python standard
library imports allowed by the hosted sandbox are also available; training
libraries and external data files are not part of the challenge runtime.
The starter uses a periodic rebalance strategy: deploy a symmetric
±50-tick range at step 0, then rebalance every 100 steps and hold in between.
Edit `self.width` and `self.rebalance_every` to tune it. The submission class is
named `Agent` and reads public settings from `config`.

`config` is a detached namespace of public scalar scenario parameters. It never
contains the live environment, private market seed, or simulation RNG. It includes
the public `policy_seed` used for reproducible policy randomness.

Return a finite numeric array of shape `(num_trajectories, 3)`:
`[lower_offset, upper_offset, hold_flag]`.
The first two entries are tick offsets from the current pool tick, rounded and
clipped to `[-tau, tau-1]` and `[-tau+1, tau]`; invalid ordering is corrected by
the engine. A positive hold flag keeps the existing position. A nonpositive
flag deploys/rebalances. Your agent makes a decision every simulation step.

Each path can hold at most one LP position. Deployment invests all available
wealth; rebalancing collects accrued fees and reinvests all remaining wealth
after costs. Holding before the first deployment keeps the initial capital in
token1 cash. There is no separate withdrawal-to-cash action: once deployed,
holding keeps the existing position and rebalancing replaces it.

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

Local practice and hosted evaluation use the following fixed market parameters.

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

The simulator uses an Euler midprice update, exact Poisson probability
`1-exp(-intensity*dt)` with at most one arrival per side per step, directional
liquidity-depth impact with stochastic tick rounding, and fee allocation.
The first deployment is free; subsequent rebalances pay gas. Never deploying
keeps cash and has zero reward.

Each step's reward is `V_next - V_current - 0.4 * dt * token0_inventory_next**2`.
The **score** is the mean cumulative reward over all paths and seeds; higher
is better. **Raw PnL** is final wealth minus initial wealth and is displayed
separately. Fees and costs already enter wealth; do not deduct them twice.
The best valid submission per participant is ranked on the hosted leaderboard.

Hosted live scoring uses at least **3 private seeds (300 paths)**. After closing,
each participant's best verified live submission is automatically selected for
at least **10 separate unseen final seeds (1,000 paths)**. Each seed is run twice
with fresh agents to verify identical actions and outcomes; the second run does
not add paths or weight to the score. Final results are published together.
Strategies that fail final evaluation are not ranked; no alternate submission
is tried. Infrastructure failures are retried before publication. Equal scores
use earliest submission time, then submission ID.

Randomized strategies must be reproducible. The public `config.policy_seed` is
**0**, independent of every private market seed. Python's global `random` and
NumPy's global RNG are seeded before source execution. Seed independent RNGs
explicitly, e.g. `self.rng = np.random.default_rng(config.policy_seed)`. Avoid
unseeded generators, entropy, and wall-clock time in decisions. Local practice
reloads source and resets policy randomness for each market seed, including
module-level initialization.

Hosted feedback includes aggregate scores and error categories. Agent prints
and exception details are not returned; use the local practice evaluator to
debug your strategy.

## License and provenance

This simulator is derived from SAiFE_gym. [UPSTREAM.json](UPSTREAM.json) records
the upstream revision and source-file hashes.

Original contributions retain MIT terms; inherited mbt_gym portions retain
BSD-3-Clause terms. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
