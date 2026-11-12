"""Differential checks against the original dense fee formula."""
from dataclasses import replace
from types import MethodType

import numpy as np
import pytest

from clmm_challenge.challenge import ScenarioConfig, create_environment
from clmm_challenge.rewards import PnL


def dense_fees(self, state=None, *, include_shares=True):
    """Pre-optimization formula, retained as an independent numerical oracle."""
    state = self.state if state is None else state
    ticks = self.tick_lower_global + np.arange(self.num_ticks)
    in_range = ((ticks[None, :] >= state["lp_tick_lower"][:, None]) &
                (ticks[None, :] < state["lp_tick_upper"][:, None]))
    liquidity = state["liquidity_array"]
    safe = np.where(liquidity > 0, liquidity, 1.0)
    shares = np.where(liquidity > 0, state["lp_liquidity"][:, None] / safe, 0.0) * in_range
    return (np.sum(state["fees_0"] * shares, axis=1),
            np.sum(state["fees_1"] * shares, axis=1),
            shares if include_shares else None)


@pytest.mark.parametrize("num_ticks", [1, 7, 100, 200, 2500])
def test_fee_reduction_is_exact_at_grid_edges_and_for_empty_positions(num_ticks):
    config = ScenarioConfig(num_trajectories=8, num_ticks=num_ticks)
    env = create_environment(config, 42)
    state, _ = env.reset()
    md = env.model_dynamics
    rng = np.random.default_rng(1234)
    state["lp_liquidity"][:] = rng.uniform(0, 1e5, 8)
    state["lp_liquidity"][0] = 0
    state["liquidity_array"][:] = rng.uniform(0, 1e6, (8, num_ticks))
    state["liquidity_array"][:, ::3] = 0
    for key in ("fees_0", "fees_1"):
        # Wide magnitudes catch changes in floating-point summation order.
        state[key][:] = 10.0 ** rng.uniform(-10, 10, (8, num_ticks))
    state["lp_tick_lower"][:] = md.tick_lower_global + np.array(
        [-10, -10, 0, num_ticks - 1, num_ticks, -20, 3, 0])
    state["lp_tick_upper"][:] = md.tick_lower_global + np.array(
        [num_ticks + 10, 2, num_ticks, num_ticks + 10, num_ticks + 20, -1, 3, num_ticks])
    before = {key: value.copy() for key, value in state.items()}
    expected = dense_fees(md, state)
    for include_shares in (False, True):
        actual = md._compute_gross_lp_fees(state, include_shares=include_shares)
        for i in range(3 if include_shares else 2):
            np.testing.assert_array_equal(actual[i], expected[i], strict=True)
        if not include_shares:
            assert actual[2] is None
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


@pytest.mark.parametrize("seed", [42, 314, 2718])
def test_dense_fee_reference_matches_every_state_and_reward(seed):
    config = ScenarioConfig(num_trajectories=8, n_steps=160, terminal_time=0.016)
    actual = create_environment(config, seed)
    reference = create_environment(config, seed)
    reference.model_dynamics._compute_gross_lp_fees = MethodType(dense_fees, reference.model_dynamics)
    actual.reset()
    reference.reset()
    rng = np.random.default_rng(123)
    for step in range(config.n_steps):
        action = np.column_stack((rng.integers(-50, 0, 8), rng.integers(1, 51, 8),
                                  np.where((step + np.arange(8)) % 11 == 0, -1, 1)))
        left = actual.step(action)
        right = reference.step(action)
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key], err_msg=f"{step}: {key}")
        for index in (1, 2, 3):
            np.testing.assert_array_equal(left[index], right[index])
        for key in left[4]:
            np.testing.assert_array_equal(left[4][key], right[4][key])
    # No cached position or fee data survives reset.
    left, _ = actual.reset()
    right, _ = reference.reset()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_explicit_refresh_and_portfolio_queries_recompute_mutated_fees():
    config = replace(ScenarioConfig(), num_trajectories=2, volatility=0,
                     arrival_floor=0, arrival_rate=0, arbitrage_coefficient=0)
    env = create_environment(config, 42)
    state, _ = env.reset()
    env.step(np.tile([-50, 50, -1], (2, 1)))
    wealth = state["portfolio_value"].copy()
    state["fees_0"][:] = 0.3
    state["fees_1"][:] = 0.7
    expected = env.model_dynamics.compute_portfolio_value(state)
    assert np.all(expected > wealth)
    env._compute_derived_obs()
    np.testing.assert_array_equal(state["portfolio_value"], expected)
    np.testing.assert_array_equal(state["lp_unclaimed_fees_0"], dense_fees(env.model_dynamics)[0])
    env.reset()
    np.testing.assert_array_equal(env.state["portfolio_value"], config.initial_wealth)


def test_custom_pnl_subclass_still_receives_a_detached_full_previous_state():
    class InspectingReward(PnL):
        def calculate(self, current_state, action, next_state, is_terminal_step=False):
            assert current_state.keys() == next_state.keys()
            for key in current_state:
                assert not np.shares_memory(current_state[key], next_state[key]), key
            np.testing.assert_array_equal(current_state["lp_ever_deployed"], False)
            np.testing.assert_array_equal(next_state["lp_ever_deployed"], True)
            return super().calculate(current_state, action, next_state, is_terminal_step)

    config = ScenarioConfig(num_trajectories=2)
    env = create_environment(config, 42)
    env.reward_function = InspectingReward()
    env.reset()
    state, rewards, *_ = env.step(np.tile([-50, 50, -1], (2, 1)))
    np.testing.assert_array_equal(rewards, state["portfolio_value"] - config.initial_wealth)
