"""Selected upstream regressions; only imports and test selection changed."""


import numpy as np


import pytest


from clmm_challenge.index_names import (
    ASSET_PRICE_KEY,
    FEES0_KEY,
    FEES1_KEY,
    LP_COLLECTED_FEES0_KEY,
    LP_COLLECTED_FEES1_KEY,
    LP_EVER_DEPLOYED_KEY,
    LP_FEE_SNAPSHOT0_KEY,
    LP_FEE_SNAPSHOT1_KEY,
    LP_LIQUIDITY_KEY,
    LP_TICK_LOWER_KEY,
    LP_TICK_UPPER_KEY,
    LP_UNCLAIMED_FEES0_KEY,
    LP_UNCLAIMED_FEES1_KEY,
    INITIAL_WEALTH_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)


from clmm_challenge.fees import UniswapV3FeeAccounting


from clmm_challenge.impact import (
    LiquidityDepthUniswapV3PriceImpact,
    SwapResult,
)


EXPONENTIAL_VALUE = 1.0001


def _make_state(
    num_trajectories=1,
    num_ticks=10,
    tick_lower_global=100,
    current_tick=105,
    asset_price=100.0,
):
    sqrt_grid = np.sqrt(
        EXPONENTIAL_VALUE ** (tick_lower_global + np.arange(num_ticks + 1, dtype=np.float64))
    )
    current_tick = np.full(num_trajectories, current_tick, dtype=np.float64)
    return {
        POOL_CURRENT_TICK_KEY: current_tick,
        POOL_SQRT_PRICE_KEY: sqrt_grid[current_tick.astype(np.int64) - tick_lower_global],
        POOL_LIQUIDITY_ARRAY_KEY: np.full((num_trajectories, num_ticks), 1e6, dtype=np.float64),
        FEES0_KEY: np.zeros((num_trajectories, num_ticks), dtype=np.float64),
        FEES1_KEY: np.zeros((num_trajectories, num_ticks), dtype=np.float64),
        ASSET_PRICE_KEY: np.full(num_trajectories, asset_price, dtype=np.float64),
    }, sqrt_grid


def _buy_capacity(state, sqrt_grid, trajectory, fee_idx):
    liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, fee_idx]
    return liquidity * (sqrt_grid[fee_idx + 1] - sqrt_grid[fee_idx])


def _sell_capacity(state, sqrt_grid, trajectory, fee_idx):
    liquidity = state[POOL_LIQUIDITY_ARRAY_KEY][trajectory, fee_idx]
    return liquidity * (1.0 / sqrt_grid[fee_idx] - 1.0 / sqrt_grid[fee_idx + 1])


def _gross_input(curve_input, fee_multiplier):
    return curve_input * (1.0 + fee_multiplier)


def _capacity_weighted_amounts(capacities, curve_input):
    capacities = np.asarray(capacities, dtype=np.float64)
    total = capacities.sum()
    if total <= 0.0:
        return np.full(capacities.shape, curve_input / capacities.size)
    return curve_input * capacities / total


def _fixed_sampler(values):
    values = np.asarray(values, dtype=np.float64)

    def sampler(rng, size):
        if values.size == 1:
            return np.full(size, values.item(), dtype=np.float64)
        assert values.shape == (size,)
        return values.copy()

    return sampler


class TestLiquidityDepthUniswapV3PriceImpact:
    def test_buy_multi_tick_fee_allocation_is_capacity_weighted(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        first_capacity = _buy_capacity(state, sqrt_grid, 0, idx)
        state[POOL_LIQUIDITY_ARRAY_KEY][0, idx + 1] *= (
            3.0 * first_capacity / _buy_capacity(state, sqrt_grid, 0, idx + 1)
        )
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        curve_input = capacities.sum()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(curve_input),
            depth_window=2,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))
        np.testing.assert_allclose(result.amounts, np.array([0.25, 0.75]) * curve_input)


    def test_sell_multi_tick_fee_allocation_is_capacity_weighted(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        first_capacity = _sell_capacity(state, sqrt_grid, 0, idx - 1)
        state[POOL_LIQUIDITY_ARRAY_KEY][0, idx - 2] *= (
            3.0 * first_capacity / _sell_capacity(state, sqrt_grid, 0, idx - 2)
        )
        capacities = np.array([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        curve_input = capacities.sum()
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(curve_input),
            depth_window=2,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 103
        np.testing.assert_array_equal(result.fee_indices, np.array([idx - 1, idx - 2]))
        np.testing.assert_allclose(result.amounts, np.array([0.25, 0.75]) * curve_input)


    def test_fractional_eta_uses_seeded_stochastic_rounding(self):
        state, sqrt_grid = _make_state(tick_lower_global=100, current_tick=105)
        idx = 105 - 100
        depth = _buy_capacity(state, sqrt_grid, 0, idx)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(1.999 * depth),
            depth_window=1,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 107
        capacities = np.array([
            _buy_capacity(state, sqrt_grid, 0, idx),
            _buy_capacity(state, sqrt_grid, 0, idx + 1),
        ])
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx + 1]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 1.999 * depth),
        )


    @pytest.mark.parametrize("direction", [1, -1], ids=["buy", "sell"])
    @pytest.mark.parametrize("zero_liquidity", [False, True], ids=["nonuniform", "zero"])
    def test_mixed_trajectories_at_array_boundary(self, direction, zero_liquidity):
        num_ticks = 10
        tick_lower_global = 100
        fee_multiplier = 0.25
        state, sqrt_grid = _make_state(num_trajectories=5)
        # Trajectories 0 and 2 move one and five ticks, respectively; the others
        # are inactive, have a zero-sized order, or are already at the boundary.
        indices = np.array([9, 7, 5, 9, 10])
        if direction == -1:
            indices = num_ticks - indices
        state[POOL_CURRENT_TICK_KEY] = tick_lower_global + indices
        state[POOL_SQRT_PRICE_KEY] = sqrt_grid[indices]
        state[POOL_LIQUIDITY_ARRAY_KEY] = (
            1e6 * np.arange(1, 6)[:, None] * np.arange(1, num_ticks + 1)[None, :]
        )
        if zero_liquidity:
            state[POOL_LIQUIDITY_ARRAY_KEY].fill(0.0)
        liquidity_before = state[POOL_LIQUIDITY_ARRAY_KEY].copy()
        gross_inputs = np.array([1e9, 2e9, 0.0, 3e9])
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_inputs),
            depth_window=3,
            num_trajectories=5,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True, False, True, True, True]),
            direction,
            tick_lower_global=tick_lower_global,
            sqrt_grid=sqrt_grid,
            num_ticks=num_ticks,
            fee_multiplier=fee_multiplier,
        )

        expected_indices = indices + direction * np.array([1, 0, 5, 0, 0])
        np.testing.assert_array_equal(
            state[POOL_CURRENT_TICK_KEY], tick_lower_global + expected_indices
        )
        np.testing.assert_array_equal(state[POOL_SQRT_PRICE_KEY], sqrt_grid[expected_indices])
        np.testing.assert_array_equal(state[POOL_LIQUIDITY_ARRAY_KEY], liquidity_before)
        np.testing.assert_array_equal(state[FEES0_KEY], 0.0)
        np.testing.assert_array_equal(state[FEES1_KEY], 0.0)

        expected_traj = np.array([0, 2, 2, 2, 2, 2])
        if direction == 1:
            expected_fee_indices = np.array([9, 5, 6, 7, 8, 9])
            capacities = _buy_capacity(state, sqrt_grid, 2, expected_fee_indices[1:])
            fee_key, other_fee_key = FEES1_KEY, FEES0_KEY
        else:
            expected_fee_indices = np.array([0, 4, 3, 2, 1, 0])
            capacities = _sell_capacity(state, sqrt_grid, 2, expected_fee_indices[1:])
            fee_key, other_fee_key = FEES0_KEY, FEES1_KEY
        curve_inputs = gross_inputs[:2] / (1.0 + fee_multiplier)
        expected_amounts = np.concatenate([
            curve_inputs[:1],
            _capacity_weighted_amounts(capacities, curve_inputs[1]),
        ])
        assert result.fee_key == fee_key
        assert result.direction == direction
        np.testing.assert_array_equal(result.trajectories, expected_traj)
        np.testing.assert_array_equal(result.fee_indices, expected_fee_indices)
        np.testing.assert_allclose(result.amounts, expected_amounts)
        np.testing.assert_allclose(
            np.bincount(result.trajectories, weights=result.amounts, minlength=5),
            np.array([curve_inputs[0], 0.0, curve_inputs[1], 0.0, 0.0]),
        )

        UniswapV3FeeAccounting().apply_fees(state, result, fee_multiplier)
        expected_fees = np.zeros_like(state[fee_key])
        expected_fees[expected_traj, expected_fee_indices] = fee_multiplier * expected_amounts
        np.testing.assert_allclose(state[fee_key], expected_fees)
        np.testing.assert_array_equal(state[other_fee_key], 0.0)


    def test_sampled_total_fee_is_gross_fee_for_zero_one_and_multi_tick_jumps(self):
        fee_tier = 0.003
        fee_multiplier = fee_tier / (1.0 - fee_tier)
        state, sqrt_grid = _make_state(
            num_trajectories=3,
            tick_lower_global=100,
            current_tick=105,
        )
        idx = 105 - 100
        depth = _buy_capacity(state, sqrt_grid, 0, idx)
        gross_inputs = np.array([0.1 * depth, 1.2 * depth, 2.2 * depth])
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(gross_inputs),
            depth_window=1,
            num_trajectories=3,
            seed=7,
        )

        result = model.process_swap(
            state,
            np.array([True, True, True]),
            1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
            fee_multiplier=fee_multiplier,
        )
        UniswapV3FeeAccounting().apply_fees(state, result, fee_multiplier)

        np.testing.assert_array_equal(result.trajectories, np.array([0, 1, 2, 2]))
        np.testing.assert_array_equal(result.fee_indices, np.array([idx, idx, idx, idx + 1]))
        np.testing.assert_allclose(
            state[FEES1_KEY].sum(axis=1),
            fee_tier * gross_inputs,
        )
        np.testing.assert_allclose(
            np.array([
                result.amounts[result.trajectories == i].sum()
                for i in range(3)
            ]),
            gross_inputs / (1.0 + fee_multiplier),
        )
        assert np.sum(state[FEES0_KEY]) == 0.0


    def test_token1_notional_sell_size_is_converted_to_token0_using_external_midprice(self):
        external_midprice = 150.0
        state, sqrt_grid = _make_state(
            tick_lower_global=100,
            current_tick=105,
            asset_price=external_midprice,
        )
        idx = 105 - 100
        average_token0_depth = np.mean([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        pool_price = state[POOL_SQRT_PRICE_KEY][0] ** 2
        assert not np.isclose(pool_price, external_midprice)
        model = LiquidityDepthUniswapV3PriceImpact(
            trade_size_sampler=_fixed_sampler(2.0 * average_token0_depth * external_midprice),
            depth_window=2,
            trade_size_unit="token1_notional",
        )

        result = model.process_swap(
            state,
            np.array([True]),
            -1,
            tick_lower_global=100,
            sqrt_grid=sqrt_grid,
            num_ticks=10,
        )

        assert state[POOL_CURRENT_TICK_KEY][0] == 103
        capacities = np.array([
            _sell_capacity(state, sqrt_grid, 0, idx - 1),
            _sell_capacity(state, sqrt_grid, 0, idx - 2),
        ])
        np.testing.assert_array_equal(result.fee_indices, np.array([idx - 1, idx - 2]))
        np.testing.assert_allclose(
            result.amounts,
            _capacity_weighted_amounts(capacities, 2.0 * average_token0_depth),
        )


class TestUniswapV3FeeAccounting:
    def test_none_swap_result_is_noop(self):
        state, _ = _make_state(num_trajectories=2)
        before = {key: value.copy() for key, value in state.items()}
        model = UniswapV3FeeAccounting()

        model.apply_fees(state, None, fee_multiplier=0.003)

        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)

    def test_sell_result_updates_fees0(self):
        state, _ = _make_state(num_trajectories=2)
        model = UniswapV3FeeAccounting()
        result = SwapResult(
            trajectories=np.array([0, 1]),
            fee_key=FEES0_KEY,
            fee_indices=np.array([3, 4]),
            amounts=np.array([10.0, 20.0]),
            direction=-1,
        )

        model.apply_fees(state, result, fee_multiplier=0.003)

        assert state[FEES0_KEY][0, 3] == pytest.approx(0.03)
        assert state[FEES0_KEY][1, 4] == pytest.approx(0.06)
        assert np.sum(state[FEES1_KEY]) == 0.0

    def test_buy_result_updates_fees1(self):
        state, _ = _make_state(num_trajectories=3)
        model = UniswapV3FeeAccounting()
        result = SwapResult(
            trajectories=np.array([1, 2]),
            fee_key=FEES1_KEY,
            fee_indices=np.array([5, 6]),
            amounts=np.array([30.0, 40.0]),
            direction=1,
        )

        model.apply_fees(state, result, fee_multiplier=0.003)

        assert np.sum(state[FEES0_KEY]) == 0.0
        assert state[FEES1_KEY][1, 5] == pytest.approx(0.09)
        assert state[FEES1_KEY][2, 6] == pytest.approx(0.12)

    def test_repeated_fee_indices_are_scatter_added(self):
        state, _ = _make_state(num_trajectories=1)
        model = UniswapV3FeeAccounting()
        result = SwapResult(
            trajectories=np.array([0, 0, 0]),
            fee_key=FEES0_KEY,
            fee_indices=np.array([3, 3, 4]),
            amounts=np.array([10.0, 20.0, 40.0]),
            direction=-1,
        )

        model.apply_fees(state, result, fee_multiplier=0.003)

        assert state[FEES0_KEY][0, 3] == pytest.approx(0.09)
        assert state[FEES0_KEY][0, 4] == pytest.approx(0.12)
        assert np.sum(state[FEES1_KEY]) == 0.0
