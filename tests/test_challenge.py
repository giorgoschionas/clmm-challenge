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


def test_cash_and_inventory_formula(config):
    assert evaluate_agent(CashAgent, config, [42, 43])["score"] == 0
    env = create_environment(config, 42)
    state, _ = env.reset()
    penalty, score = np.zeros(3), np.zeros(3)
    agent = DeployOnceAgent(config.submission_namespace())
    for _ in range(config.n_steps):
        state, reward, *_ = env.step(agent.get_action(official_observation(state, env)))
        penalty += config.inventory_phi * config.step_size * state["lp_token0_amount"] ** 2
        score += reward
    np.testing.assert_allclose(score, state["portfolio_value"] - config.initial_wealth - penalty, atol=1e-10)


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
    assert len(obs) == 12
    assert all(v.shape == (n,) for v in obs.values())
    obs["portfolio_value"][:] = -1
    assert np.all(state["portfolio_value"] == c.initial_wealth)
    assert not {"seed", "rng", "liquidity_array", "fees_0", "fees_1"} & obs.keys()


def test_free_deployment_then_gas_and_exhaustion(config):
    c = replace(config, volatility=0, arrival_floor=0, arrival_rate=0,
                arbitrage_coefficient=0, inventory_phi=0)
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
    assert result["score"] <= result["mean_pnl"]
    assert result["pnl_pct"] == pytest.approx(result["mean_pnl"] / 10)


def test_invalid_source_and_renamed_agent(tmp_path):
    p = tmp_path / "agent.py"
    p.write_text("class Agent: pass")
    with pytest.raises(ValueError):
        load_agent(p)
    p.write_text("from concentrator import Agent\nclass MyAgent(Agent):\n def __init__(self, config): pass\n def get_action(self, state): return None\n")
    assert load_agent(p).__name__ == "MyAgent"
