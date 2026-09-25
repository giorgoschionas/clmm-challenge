"""Fixed CLMM v1 scenario and the public participant interface."""
from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np

from .AMMEnvironment import AMMEnvironment
from .ModelDynamics import UniswapV3ModelDynamics
from .arrivals import LiquidityKernelArrivalModel
from .midprice import GeometricBrownianMotionMidpriceModel
from .impact import LiquidityDepthUniswapV3PriceImpact
from .fees import UniswapV3FeeAccounting
from .rewards import RunningInventoryPenalty

ENGINE_ID = "clmm-v1"
PRACTICE_SEEDS = (42, 314, 2718)
OBSERVATION_KEYS = (
    "sqrt_price", "current_tick", "midprice", "time", "lp_tick_lower",
    "lp_tick_upper", "lp_ever_deployed", "gas_cost", "portfolio_value",
    "lp_token0_amount", "lp_alpha",
)


@dataclass(frozen=True)
class ScenarioConfig:
    terminal_time: float = 1.0
    n_steps: int = 1000
    num_trajectories: int = 100
    initial_wealth: float = 1000.0
    initial_price: float = 100.0
    tau: int = 50
    num_ticks: int = 7000
    exponential_value: float = 1.0001
    drift: float = 0.0
    volatility: float = 0.03
    arrival_floor: float = 10.0
    arrival_rate: float = 300.0
    liquidity_coefficient: float = 0.0
    arbitrage_coefficient: float = 4000.0
    kernel_beta: float = 0.5
    kernel_window: int = 10
    liquidity_scale: float = 1e6
    trade_notional: float = 250.0
    depth_window: int = 10
    min_depth: float = 1e-12
    fee_tier: float = 0.003
    gas_cost: float = 2.0
    swap_fee_rate: float = 0.0
    inventory_phi: float = 0.4
    inventory_exponent: float = 2.0
    terminal_inventory_penalty: float = 0.0

    @property
    def step_size(self):
        return self.terminal_time / self.n_steps

    def submission_namespace(self):
        # A detached scalar snapshot, not a handle to the environment or RNG.
        return SimpleNamespace(**{f.name: getattr(self, f.name) for f in fields(self)},
                               step_size=self.step_size)


def create_environment(config: ScenarioConfig, seed: int) -> AMMEnvironment:
    c = config
    midprice = GeometricBrownianMotionMidpriceModel(
        drift=c.drift, volatility=c.volatility, initial_price=c.initial_price,
        terminal_time=c.terminal_time, step_size=c.step_size,
        num_trajectories=c.num_trajectories, seed=seed,
    )
    arrivals = LiquidityKernelArrivalModel(
        alpha=np.repeat(np.array([c.arrival_floor, c.arrival_rate,
                                  c.liquidity_coefficient, c.arbitrage_coefficient])[:, None], 2, axis=1),
        beta=c.kernel_beta, K=c.kernel_window, liquidity_scale=c.liquidity_scale,
        step_size=c.step_size, num_trajectories=c.num_trajectories, seed=seed + 1,
    )
    impact = LiquidityDepthUniswapV3PriceImpact(
        trade_size_sampler=lambda rng, size: np.full(size, c.trade_notional, dtype=np.float64),
        depth_window=c.depth_window, min_depth=c.min_depth, trade_size_unit="token1_notional",
        num_trajectories=c.num_trajectories, seed=seed + 2,
    )
    dynamics = UniswapV3ModelDynamics(
        midprice_model=midprice, arrival_model=arrivals, price_impact_model=impact,
        fee_accounting_model=UniswapV3FeeAccounting(), num_trajectories=c.num_trajectories,
        fee_tier=c.fee_tier, tau=c.tau, num_ticks=c.num_ticks,
        exponential_value=c.exponential_value, gas_cost=c.gas_cost,
        swap_fee_rate=c.swap_fee_rate, seed=seed + 3,
    )
    return AMMEnvironment(
        terminal_time=c.terminal_time, n_steps=c.n_steps, initial_wealth=c.initial_wealth,
        num_trajectories=c.num_trajectories, model_dynamics=dynamics, seed=seed,
        reward_function=RunningInventoryPenalty(
            per_step_inventory_aversion=c.inventory_phi,
            terminal_inventory_aversion=c.terminal_inventory_penalty,
            inventory_exponent=c.inventory_exponent,
        ),
    )


def official_observation(state: dict, env: AMMEnvironment) -> dict:
    observation = {key: np.asarray(state[key]).copy() for key in OBSERVATION_KEYS}
    md = env.model_dynamics
    tick = np.clip(state["current_tick"] - md.tick_lower_global, 0, md.num_ticks - 1)
    observation["active_liquidity"] = state["liquidity_array"][np.arange(env.num_trajectories), tick].copy()
    return observation


def validate_action(action, num_trajectories: int):
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("action must be numeric") from exc
    if array.shape != (num_trajectories, 3) or not np.isfinite(array).all():
        raise ValueError(f"action must contain finite values with shape ({num_trajectories}, 3)")
    return array
