"""Optional mechanics and nominal-configuration audit against upstream."""
import ast
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from clmm_challenge import challenge


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("seed", [42, 314, 2718])
def test_full_state_and_reward_parity(monkeypatch, n, seed):
    root = os.environ.get("SAIFE_UPSTREAM_PATH")
    if not root:
        pytest.skip("set SAIFE_UPSTREAM_PATH to audit against upstream")
    monkeypatch.syspath_prepend(str(Path(root).resolve()))
    sources = {
        "AMMEnvironment": "gym.AMMEnvironment",
        "UniswapV3ModelDynamics": "gym.ModelDynamics",
        "LiquidityKernelArrivalModel": "stochastic_processes.arrival_models",
        "GeometricBrownianMotionMidpriceModel": "stochastic_processes.midprice_models",
        "LiquidityDepthUniswapV3PriceImpact": "stochastic_processes.price_impact_models",
        "UniswapV3FeeAccounting": "stochastic_processes.fee_accounting_models",
        "PnL": "rewards.RewardFunctions",
    }
    config = challenge.ScenarioConfig(num_trajectories=n, n_steps=30,
                                      terminal_time=30 * challenge.ScenarioConfig().step_size,
                                      num_ticks=200)
    extracted = challenge.create_environment(config, seed)
    with monkeypatch.context() as patch:
        for cls, module in sources.items():
            patch.setattr(challenge, cls, getattr(importlib.import_module("SAiFE_gym." + module), cls))
        original = challenge.create_environment(config, seed)
    left, _ = extracted.reset()
    right, _ = original.reset()
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_allclose(left[key], right[key], rtol=0, atol=1e-11, err_msg=key)
    rng = np.random.default_rng(123)
    for step in range(config.n_steps):
        action = np.column_stack([rng.integers(-50, 0, n), rng.integers(1, 51, n),
                                  rng.choice([-1, 1], n)])
        # Mix held and rebalanced paths, including zero-cost initial deployment.
        a = extracted.step(action)
        b = original.step(action)
        assert a[0].keys() == b[0].keys()
        for key in a[0]:
            np.testing.assert_allclose(a[0][key], b[0][key], rtol=0, atol=1e-11, err_msg=key)
        for i in (1, 2, 3):
            np.testing.assert_allclose(a[i], b[i], rtol=0, atol=1e-11)
        np.testing.assert_array_equal(extracted.model_dynamics.last_arrivals,
                                      original.model_dynamics.last_arrivals)
    assert a[2].all()


def test_nominal_market_defaults_match_upstream_trainer():
    root = os.environ.get("SAIFE_UPSTREAM_PATH")
    if not root:
        pytest.skip("set SAIFE_UPSTREAM_PATH to audit against upstream")
    root = Path(root)
    # Read the CLI defaults without importing PPO, Torch, or plotting libraries
    # into the standalone package's deliberately small runtime.
    helpers = ast.parse((root / "experiments/helpers.py").read_text())
    constants = {
        node.targets[0].id: node.value for node in helpers.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    trainer = ast.parse((root / "experiments/train_robust_lp_agent.py").read_text())
    parser = next(node for node in trainer.body
                  if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    arguments = {
        ast.literal_eval(node.args[0]): next(kw.value for kw in node.keywords if kw.arg == "default")
        for node in ast.walk(parser)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and any(kw.arg == "default" for kw in node.keywords)
    }
    c = challenge.ScenarioConfig()
    for field, flag in {
        # Step count is a challenge-specific resolution override; the horizon
        # and stochastic market parameters still match the nominal trainer.
        "terminal_time": "--terminal-time",
        "num_trajectories": "--num-trajectories", "initial_wealth": "--initial-wealth",
        "background_liquidity": "--background-liquidity", "tau": "--tau",
        "volatility": "--nominal-sigma", "arrival_rate": "--nominal-arrival-rate",
        "liquidity_coefficient": "--arrival-alpha2", "arbitrage_coefficient": "--alpha3",
        "kernel_beta": "--kernel-beta", "kernel_window": "--kernel-window",
        "liquidity_scale": "--liquidity-scale", "trade_notional": "--trade-size-notional",
        "depth_window": "--price-impact-depth-window", "min_depth": "--price-impact-min-depth",
        "gas_cost": "--nominal-gas-cost",
    }.items():
        default = arguments[flag]
        if isinstance(default, ast.Name):
            default = constants[default.id]
        assert getattr(c, field) == ast.literal_eval(default), field

    for field, name in {"terminal_time": "TERMINAL_TIME", "initial_price": "INITIAL_PRICE",
                        "fee_tier": "FEE_TIER", "num_ticks": "NUM_TICKS"}.items():
        assert getattr(c, field) == ast.literal_eval(constants[name]), field
