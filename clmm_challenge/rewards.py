import abc
from typing import Union

import numpy as np
from clmm_challenge.index_names import PORTFOLIO_VALUE_KEY


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: Union[np.ndarray, dict], action: np.ndarray,
        next_state: Union[np.ndarray, dict], is_terminal_step: bool = False
    ) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass


# ---------------------------------------------------------------------------
# Concrete reward functions
# ---------------------------------------------------------------------------

class PnL(RewardFunction):
    """Mark-to-market PnL reward: change in LP portfolio value between steps.

    Reads `PORTFOLIO_VALUE_KEY` from state, which the env computes in
    `AMMEnvironment._compute_derived_obs`.
    """

    def calculate(
        self, current_state: dict, action: np.ndarray,
        next_state: dict, is_terminal_step: bool = False
    ) -> np.ndarray:
        return next_state[PORTFOLIO_VALUE_KEY] - current_state[PORTFOLIO_VALUE_KEY]

    def reset(self, initial_state: Union[np.ndarray, dict]):
        pass
