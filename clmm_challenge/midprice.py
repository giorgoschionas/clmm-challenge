from typing import Optional
from math import sqrt

from clmm_challenge.processes import StochasticProcessModel

import numpy as np


class MidpriceModel(StochasticProcessModel):
    """Base class for vectorized midprice processes.

    ``volatility`` remains the scalar constructor/configuration value. Domain
    randomization uses ``episode_volatility`` so every trajectory can carry a
    different value without changing the public scalar API.
    """

    def set_episode_volatility(self, volatility: np.ndarray) -> None:
        episode_volatility = np.asarray(volatility, dtype=np.float64)
        expected_shape = (self.num_trajectories, 1)
        if episode_volatility.shape != expected_shape:
            raise ValueError(
                "episode volatility must have shape "
                f"{expected_shape}, got {episode_volatility.shape}"
            )
        if not np.all(np.isfinite(episode_volatility)):
            raise ValueError("episode volatility must contain only finite values")
        if np.any(episode_volatility < 0.0):
            raise ValueError("episode volatility must be non-negative")
        self.episode_volatility = episode_volatility.copy()


def _initialize_episode_volatility(model: MidpriceModel) -> None:
    model.set_episode_volatility(
        np.full(
            (model.num_trajectories, 1),
            model.volatility,
            dtype=np.float64,
        )
    )



class GeometricBrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = float(volatility)
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )
        _initialize_episode_volatility(self)

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.current_state * self.step_size
            + self.episode_volatility
            * self.current_state
            * sqrt(self.step_size)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        stdev = sqrt(
            initial_price**2
            * np.exp(2 * self.drift * terminal_time)
            * (np.exp(self.volatility**2 * terminal_time) - 1)
        )
        return initial_price * np.exp(self.drift * terminal_time) + 4 * stdev
