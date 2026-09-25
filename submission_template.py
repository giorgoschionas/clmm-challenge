import numpy as np


class Agent:
    """Deploy a wide position once, then hold. Improve this baseline!"""

    def __init__(self, config):
        self.tau = config.tau

    def get_action(self, state):
        n = state["sqrt_price"].shape[0]
        return np.column_stack([
            np.full(n, -self.tau), np.full(n, self.tau),
            np.where(state["lp_ever_deployed"], 1.0, -1.0),
        ])
