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
| `lp_token0_amount` | Token0 inventory in the LP position |
| `lp_alpha` | LP token0 value fraction |

## Fixed v1 market and scoring

Local practice and the hosted engine use the following fixed market parameters,
based on the nominal market in SAiFE_gym's `experiments/train_robust_lp_agent.py`.
The challenge uses a finer 10,000-step resolution over the same horizon of 1.0,
so each step advances time by 0.0001.
The tracked liquidity grid contains 2,500 tick intervals, centered on the initial
pool tick, to reduce the arrays processed each step.
Volatility and baseline arrival rate stay fixed throughout evaluation. The
challenge uses plain PnL as its reward.

| Parameter | Value |
|---|---|
| Paths / steps / horizon | 100 / 10,000 / 1.0 |
| Initial price / wealth | 100 / 1,000 token1 |
| Midprice | Geometric Brownian model; drift 0, volatility 0.03 |
| Arrival coefficients per side | Floor 10, baseline 300, liquidity 0, arbitrage 4,000 |
| Liquidity kernel | Beta 0.5, window 10, normalization 1,000,000 |
| Price impact | Liquidity-depth; fixed gross trade notional 250 token1 |
| Depth window / minimum depth | 15 / 1e-12 |
| Ticks / spacing / background liquidity | 2,500 / 1.0001 / 30,000 per tick interval |
| LP offset limit | ±50 ticks |
| Pool fee | 0.003 |
| Gas / additional rebalance swap fee | 2 / 0 |
| Reward | Change in portfolio value (PnL); no inventory penalty |

Background liquidity is uniform across the tracked intervals, separate from the
agent's capital, and restored on reset. It is measured in Uniswap-style liquidity
units, not token1 cash. The agent starts each path with 1,000 token1 in cash and
adds its own liquidity inside its chosen range when it deploys. The public
`config.background_liquidity` reports the background amount.

The simulator uses an Euler midprice update, exact Poisson probability
`1-exp(-intensity*dt)` with at most one arrival per side per step, directional
liquidity-depth impact with stochastic tick rounding, and fee allocation.
Trades consume interval capacities in execution order; the depth window controls
how many intervals are inspected at once, rather than capping the tick movement.
The first deployment is free; subsequent rebalances pay gas. Never deploying
keeps cash and has zero reward.

Each step's reward is `V_next - V_current`. Cumulative reward is therefore final
wealth minus initial wealth. The **score** is mean PnL in token1 over all paths
and seeds; higher is better. **PnL percentage** is `100 * score / initial_wealth`.
Fees and costs already enter wealth; do not deduct them twice. There is no
running or terminal inventory penalty.
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
the original extraction and the later nominal-configuration/PnL update, with
upstream revisions, source-file hashes, and challenge-specific overrides. To
check full state/reward parity and the shared market defaults against that
upstream checkout, run:

```bash
SAIFE_UPSTREAM_PATH=/path/to/SAiFE_gym .venv/bin/python -m pytest
```

Hosted suites freeze both source and configuration fingerprints. This update
changes that fingerprint; previously registered suites and historical scores
belong to the earlier configuration and must not be mixed with this one.

Original contributions retain MIT terms; inherited mbt_gym portions retain
BSD-3-Clause terms. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
