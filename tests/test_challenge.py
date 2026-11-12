from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clmm_challenge.challenge import ScenarioConfig, create_environment, official_observation, validate_action
from clmm_challenge.baselines import CashAgent, DeployOnceAgent
from clmm_challenge.evaluate import evaluate_agent, load_agent, run_episode


@pytest.fixture
def config():
    return ScenarioConfig(num_trajectories=3, n_steps=20, terminal_time=0.02, num_ticks=200)


def test_default_resolution_reaches_terminal_horizon():
    c = replace(ScenarioConfig(), num_trajectories=1, num_ticks=200)
    assert c.n_steps == 10_000
    assert c.terminal_time == 1.0
    assert c.step_size == pytest.approx(0.0001)
    env = create_environment(c, 42)
    assert env.model_dynamics.midprice_model.step_size == c.step_size
    assert env.model_dynamics.arrival_model.step_size == c.step_size
    state, _ = env.reset()
    action = np.array([[-50., 50., 1.]])
    for step in range(c.n_steps):
        state, reward, terminated, truncated, _ = env.step(action)
        assert bool(terminated.all()) == (step == c.n_steps - 1)
        assert not truncated.any()
        np.testing.assert_array_equal(reward, 0)
    np.testing.assert_allclose(state["time"], c.terminal_time, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(state["portfolio_value"], c.initial_wealth)


@pytest.mark.parametrize("n", [1, 3])
def test_cash_and_pnl_reward(config, n):
    c = replace(config, num_trajectories=n)
    cash = evaluate_agent(CashAgent, c, [42, 43])
    assert cash["score"] == cash["mean_pnl"] == cash["pnl_pct"] == 0
    env = create_environment(c, 42)
    state, _ = env.reset()
    score = np.zeros(n)
    saw_inventory = False
    for step in range(c.n_steps):
        previous_wealth = state["portfolio_value"].copy()
        # Exercise deployment, holds, fees, and gas-paying rebalances.
        action = np.tile([-50, 50, -1 if step % 5 == 0 else 1], (n, 1))
        state, reward, *_ = env.step(action)
        saw_inventory |= np.any(state["lp_token0_amount"] > 0)
        np.testing.assert_allclose(reward, state["portfolio_value"] - previous_wealth, atol=1e-10)
        score += reward
    assert saw_inventory
    np.testing.assert_allclose(score, state["portfolio_value"] - c.initial_wealth, atol=1e-10)


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("background", [0.0, 12_345.0, 30_000.0, 100_000.0])
def test_background_liquidity_and_cash_restored_on_reset(config, n, background):
    c = replace(config, num_trajectories=n, background_liquidity=background,
                volatility=0, arrival_floor=0, arrival_rate=0, arbitrage_coefficient=0)
    assert c.submission_namespace().background_liquidity == background
    env = create_environment(c, 42)
    state, _ = env.reset()
    np.testing.assert_array_equal(state["liquidity_array"], background)
    np.testing.assert_array_equal(state["portfolio_value"], c.initial_wealth)
    np.testing.assert_array_equal(state["lp_token0_amount"], 0)
    np.testing.assert_array_equal(state["lp_liquidity"], 0)
    state, *_ = env.step(np.tile([-50, 50, -1], (n, 1)))
    assert np.all(np.any(state["liquidity_array"] > background, axis=1))
    state, _ = env.reset()
    np.testing.assert_array_equal(state["liquidity_array"], background)
    np.testing.assert_array_equal(state["portfolio_value"], c.initial_wealth)
    np.testing.assert_array_equal(state["lp_liquidity"], 0)


@pytest.mark.parametrize("background", [-1, np.nan, np.inf, -np.inf, [30_000], "invalid"])
def test_invalid_background_liquidity(config, background):
    with pytest.raises(ValueError, match="background_liquidity"):
        create_environment(replace(config, background_liquidity=background), 42)


@pytest.mark.parametrize("n", [1, 3])
def test_reproducibility_and_observation_isolation(config, n):
    c = replace(config, num_trajectories=n)
    env = create_environment(c, 5)
    first = run_episode(env, DeployOnceAgent(c.submission_namespace()))
    env.reset(seed=5)
    second = run_episode(env, DeployOnceAgent(c.submission_namespace()))
    np.testing.assert_array_equal(first, second)
    state, _ = env.reset(seed=5)
    obs = official_observation(state, env)
    assert len(obs) == 10
    assert all(v.shape == (n,) for v in obs.values())
    np.testing.assert_array_equal(obs["lp_token0_amount"], state["lp_token0_amount"])
    obs["portfolio_value"][:] = -1
    assert np.all(state["portfolio_value"] == c.initial_wealth)
    assert not {"seed", "rng", "liquidity_array", "fees_0", "fees_1",
                "lp_alpha", "active_liquidity"} & obs.keys()


def test_free_deployment_then_gas_and_exhaustion(config):
    c = replace(config, volatility=0, arrival_floor=0, arrival_rate=0,
                arbitrage_coefficient=0)
    env = create_environment(c, 42)
    env.reset()
    action = np.tile([-50, 50, -1], (3, 1))
    state, _, *_ = env.step(action)
    np.testing.assert_allclose(state["portfolio_value"], c.initial_wealth)
    state, _, *_ = env.step(action)
    np.testing.assert_allclose(state["portfolio_value"], c.initial_wealth - c.gas_cost)
    c = replace(c, gas_cost=2000)
    env = create_environment(c, 42)
    env.reset()
    env.step(action)
    state, *_ = env.step(action)
    assert np.all(state["portfolio_value"] == 0)


@pytest.mark.parametrize("action", [None, [1, 2, 3], [[0, 1, np.nan]], [[0, 1, np.inf]], [["x", 1, 1]]])
def test_bad_actions(action):
    with pytest.raises(ValueError):
        validate_action(action, 1)


def test_template_and_multiseed(config):
    agent = load_agent(Path(__file__).parents[1] / "submission_template.py")
    result = evaluate_agent(agent, config, [42, 43])
    assert result["num_paths"] == 6
    assert result["score"] == pytest.approx(result["mean_pnl"], abs=1e-10)
    assert result["pnl_pct"] == pytest.approx(result["mean_pnl"] / 10)


def test_invalid_source_and_renamed_agent(tmp_path):
    p = tmp_path / "agent.py"
    p.write_text("class Agent: pass")
    with pytest.raises(ValueError):
        load_agent(p)
    p.write_text("from concentrator import Agent\nclass MyAgent(Agent):\n def __init__(self, config): pass\n def get_action(self, state): return None\n")
    assert load_agent(p).__name__ == "MyAgent"
