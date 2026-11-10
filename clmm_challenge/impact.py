import abc
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from clmm_challenge.index_names import (
    ASSET_PRICE_KEY,
    FEES0_KEY,
    FEES1_KEY,
    POOL_CURRENT_TICK_KEY,
    POOL_LIQUIDITY_ARRAY_KEY,
    POOL_SQRT_PRICE_KEY,
)
from clmm_challenge.processes import StochasticProcessModel


@dataclass
class SwapResult:
    """Vectorized metadata for one swap side.

    ``amounts`` is the curve/net input amount used as the fee basis by fee
    accounting. For some impact models this equals crossed-tick capacity; for
    sampled gross-input arrivals it is the post-fee input that reaches the AMM
    curve.
    """

    trajectories: np.ndarray
    fee_key: str
    fee_indices: np.ndarray
    amounts: np.ndarray
    direction: int




class PriceImpactModel(StochasticProcessModel):
    """Base class for AMM price-impact rules.

    SAiFE price impact is an AMM pool-state transition, not a scalar execution
    price adjustment. Stateless models can use the default empty process state.
    """

    def __init__(self, num_trajectories: int = 1, seed: int = None):
        empty_state = np.empty((1, 0))
        super().__init__(
            min_value=empty_state,
            max_value=empty_state,
            step_size=None,
            terminal_time=0.0,
            initial_state=empty_state,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: dict = None):
        return self.current_state

    @abc.abstractmethod
    def process_swap(
        self,
        state: dict,
        active: np.ndarray,
        direction: int,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        num_ticks: int,
        fee_multiplier: float = 0.0,
    ) -> Optional[SwapResult]:
        pass








class LiquidityDepthUniswapV3PriceImpact(PriceImpactModel):
    """Reduced-form liquidity-depth Uniswap V3 impact model.

    The model samples an active gross trade size, converts it to the swap input
    token's units, deducts fees to get the curve input, and accumulates interval
    capacities in execution order. The remaining input divided by the next
    interval's capacity gives the probability of crossing one additional tick.
    Prices stay on the tick lattice. Zero-tick fees
    are allocated to the first executable interval; multi-tick fees are
    allocated across realized crossed intervals by directional capacity.

    ``depth_window`` is the number of intervals inspected per batch. Additional
    windows are inspected only for trades that exhaust the current window.
    ``min_depth`` floors each interval's capacity for the impact calculation,
    keeping empty or very thin intervals numerically well-defined.

    ``trade_size_unit="input_token"`` keeps the sampled size in token0 units
    for sells and token1 units for buys. ``trade_size_unit="token1_notional"``
    treats the sampled size as token1 value on both sides, converting sell sizes
    to token0 by dividing by the external midprice.
    """

    def __init__(
        self,
        trade_size_sampler: Callable[[np.random.Generator, int], np.ndarray],
        depth_window: int = 15,
        min_depth: float = 1e-12,
        trade_size_unit: str = "input_token",
        num_trajectories: int = 1,
        seed: int = None,
    ):
        assert callable(trade_size_sampler), "trade_size_sampler must be callable"
        assert depth_window >= 1, f"depth_window must be >= 1, got {depth_window}"
        assert min_depth > 0, f"min_depth must be positive, got {min_depth}"
        assert trade_size_unit in {"input_token", "token1_notional"}, (
            "trade_size_unit must be either 'input_token' or 'token1_notional', "
            f"got {trade_size_unit!r}"
        )
        self.trade_size_sampler = trade_size_sampler
        self.depth_window = depth_window
        self.min_depth = min_depth
        self.trade_size_unit = trade_size_unit
        super().__init__(num_trajectories=num_trajectories, seed=seed)

    def process_swap(
        self,
        state: dict,
        active: np.ndarray,
        direction: int,
        *,
        tick_lower_global: int,
        sqrt_grid: np.ndarray,
        num_ticks: int,
        fee_multiplier: float = 0.0,
    ) -> Optional[SwapResult]:
        if not np.any(active):
            return None
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 for sell or 1 for buy")

        current_tick = state[POOL_CURRENT_TICK_KEY].astype(np.int64)
        idx_all = current_tick - tick_lower_global
        traj = np.flatnonzero(active)
        idx = idx_all[traj]
        liquidity_array = state[POOL_LIQUIDITY_ARRAY_KEY]
        gross_multiplier = 1.0 + float(fee_multiplier)
        assert gross_multiplier > 0.0, "fee_multiplier must be greater than -1"

        if direction == -1:
            fee_key = FEES0_KEY
            first_executable = (idx > 0) & (idx <= num_ticks)
            max_tick_move = np.maximum(idx, 0)
        else:
            fee_key = FEES1_KEY
            first_executable = (idx >= 0) & (idx < num_ticks)
            max_tick_move = np.maximum(num_ticks - idx, 0)

        sampled_trade_size = np.asarray(
            self.trade_size_sampler(self.rng, len(traj)),
            dtype=np.float64,
        ).reshape(-1)
        assert sampled_trade_size.shape == (len(traj),), (
            "trade_size_sampler must return one non-negative trade size per "
            f"active trajectory, got shape {sampled_trade_size.shape}"
        )
        assert np.all(sampled_trade_size >= 0.0), (
            "trade_size_sampler must return non-negative trade sizes"
        )

        gross_trade_size = self._to_input_token_amount(
            sampled_trade_size,
            state,
            traj,
            direction,
        )
        curve_trade_size = gross_trade_size / gross_multiplier

        whole_ticks, fractional_ticks = self._cumulative_tick_depth(
            liquidity_array,
            traj,
            idx,
            curve_trade_size,
            max_tick_move,
            sqrt_grid,
            direction,
        )
        rounded_ticks = whole_ticks + (
            self.rng.uniform(size=len(traj)) < fractional_ticks
        ).astype(np.int64)
        tick_move = np.minimum(rounded_ticks, max_tick_move).astype(np.int64)

        moving = tick_move > 0
        if np.any(moving):
            new_tick = current_tick.copy()
            new_tick[traj[moving]] += direction * tick_move[moving]
            state[POOL_CURRENT_TICK_KEY] = new_tick
            state[POOL_SQRT_PRICE_KEY] = sqrt_grid[new_tick - tick_lower_global]

        executable = first_executable & (gross_trade_size > 0.0)
        if not np.any(executable):
            return None

        exec_traj = traj[executable]
        exec_idx = idx[executable]
        exec_tick_move = tick_move[executable]
        exec_curve_trade_size = curve_trade_size[executable]
        entry_counts = np.maximum(exec_tick_move, 1)
        max_entries = int(entry_counts.max())
        fee_offsets = np.arange(max_entries, dtype=np.int64)
        fee_mask = fee_offsets[None, :] < entry_counts[:, None]

        if direction == -1:
            fee_indices_matrix = exec_idx[:, None] - 1 - fee_offsets[None, :]
        else:
            fee_indices_matrix = exec_idx[:, None] + fee_offsets[None, :]

        amounts_matrix = np.zeros_like(fee_indices_matrix, dtype=np.float64)
        zero_move = exec_tick_move == 0
        amounts_matrix[zero_move, 0] = exec_curve_trade_size[zero_move]

        moving_rows = exec_tick_move > 0
        if np.any(moving_rows):
            moving_fee_indices = fee_indices_matrix[moving_rows]
            # NumPy gathers before masking, so padded intervals need safe indices.
            moving_fee_indices_safe = np.clip(moving_fee_indices, 0, num_ticks - 1)
            moving_mask = fee_mask[moving_rows]
            moving_traj = exec_traj[moving_rows]
            moving_liquidity = (
                liquidity_array[moving_traj[:, None], moving_fee_indices_safe]
                * moving_mask
            )
            moving_capacities = self._interval_capacity(
                moving_liquidity,
                moving_fee_indices_safe,
                sqrt_grid,
                direction,
            ) * moving_mask
            capacity_sums = moving_capacities.sum(axis=1)
            equal_weights = np.divide(
                moving_mask,
                exec_tick_move[moving_rows][:, None],
                dtype=np.float64,
            )
            capacity_weights = np.divide(
                moving_capacities,
                capacity_sums[:, None],
                out=np.zeros_like(moving_capacities, dtype=np.float64),
                where=capacity_sums[:, None] > 0.0,
            )
            weights = np.where(
                capacity_sums[:, None] > 0.0,
                capacity_weights,
                equal_weights,
            )
            amounts_matrix[moving_rows] = (
                exec_curve_trade_size[moving_rows, None] * weights
            )

        return SwapResult(
            trajectories=np.broadcast_to(
                exec_traj[:, None],
                fee_indices_matrix.shape,
            )[fee_mask],
            fee_key=fee_key,
            fee_indices=fee_indices_matrix[fee_mask],
            amounts=amounts_matrix[fee_mask],
            direction=direction,
        )

    def _cumulative_tick_depth(
        self,
        liquidity_array: np.ndarray,
        trajectories: np.ndarray,
        indices: np.ndarray,
        curve_input: np.ndarray,
        max_tick_move: np.ndarray,
        sqrt_grid: np.ndarray,
        direction: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find affordable whole ticks and the next tick's rounding probability.

        Work in windows across all unfinished trajectories, rather than scanning
        to the pool boundary or looping over individual ticks or trajectories.
        """
        whole_ticks = np.zeros(len(trajectories), dtype=np.int64)
        fractions = np.zeros(len(trajectories), dtype=np.float64)
        remaining = curve_input.copy()
        pending = (remaining > 0.0) & (max_tick_move > 0)
        offsets = np.arange(min(self.depth_window, int(max_tick_move.max())))
        num_ticks = liquidity_array.shape[1]

        while np.any(pending):
            rows = np.flatnonzero(pending)
            available = max_tick_move[rows] - whole_ticks[rows]
            valid = offsets[None, :] < available[:, None]
            boundary = indices[rows] + direction * whole_ticks[rows]
            if direction == 1:
                interval_indices = boundary[:, None] + offsets[None, :]
            else:
                interval_indices = boundary[:, None] - 1 - offsets[None, :]
            safe_indices = np.clip(interval_indices, 0, num_ticks - 1)
            liquidity = liquidity_array[trajectories[rows, None], safe_indices]
            capacities = self._interval_capacity(
                liquidity, safe_indices, sqrt_grid, direction
            )
            capacities = np.where(valid, np.maximum(capacities, self.min_depth), np.inf)
            cumulative = np.cumsum(capacities, axis=1)
            crossed = valid & (cumulative <= remaining[rows, None])
            counts = crossed.sum(axis=1)
            row_indices = np.arange(len(rows))
            spent = np.where(
                counts > 0, cumulative[row_indices, np.maximum(counts - 1, 0)], 0.0
            )
            remaining[rows] = np.maximum(remaining[rows] - spent, 0.0)
            whole_ticks[rows] += counts

            # Only a partially affordable interval needs stochastic rounding.
            partial = counts < valid.sum(axis=1)
            next_capacity = capacities[row_indices, np.minimum(counts, len(offsets) - 1)]
            fractions[rows] = np.where(
                partial, np.clip(remaining[rows] / next_capacity, 0.0, 1.0), 0.0
            )
            pending[rows] = (
                ~partial & (remaining[rows] > 0.0) & (whole_ticks[rows] < max_tick_move[rows])
            )

        return whole_ticks, fractions

    @staticmethod
    def _interval_capacity(
        liquidity: np.ndarray,
        interval_indices: np.ndarray,
        sqrt_grid: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        if direction == -1:
            return liquidity * (
                1.0 / sqrt_grid[interval_indices]
                - 1.0 / sqrt_grid[interval_indices + 1]
            )
        return liquidity * (
            sqrt_grid[interval_indices + 1]
            - sqrt_grid[interval_indices]
        )

    def _to_input_token_amount(
        self,
        sampled_trade_size: np.ndarray,
        state: dict,
        trajectories: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        """Convert sampled trade sizes to the swap input token's native units."""
        if self.trade_size_unit == "input_token":
            return sampled_trade_size

        if direction == 1:
            return sampled_trade_size

        external_midprice = state[ASSET_PRICE_KEY][trajectories]
        assert np.all(external_midprice > 0.0), "external midprice must be positive"
        return sampled_trade_size / external_midprice
