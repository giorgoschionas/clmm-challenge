"""Optional extraction audit against the recorded upstream checkout."""
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
        "RunningInventoryPenalty": "rewards.RewardFunctions",
    }
    config = challenge.ScenarioConfig(num_trajectories=n, n_steps=30, terminal_time=0.03, num_ticks=200)
    extracted = challenge.create_environment(config, seed)
    with monkeypatch.context() as patch:
        for cls, module in sources.items():
            patch.setattr(challenge, cls, getattr(importlib.import_module("SAiFE_gym." + module), cls))
        original = challenge.create_environment(config, seed)
    left, _ = extracted.reset()
    right, _ = original.reset()
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
