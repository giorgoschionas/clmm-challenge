import numpy as np


class Agent:
    def __init__(self, config):
        self.num_trajectories = config.num_trajectories
        self.step_size = config.step_size
        # Tune these strategy parameters; width must stay within [1, config.tau].
        self.rebalance_every = 100
        self.width = 50
        assert self.rebalance_every >= 1
        assert 1 <= self.width <= config.tau

    def get_action(self, state):
        n = self.num_trajectories
        lower = np.full(n, -self.width, dtype=np.float32)
        upper = np.full(n, self.width, dtype=np.float32)

        # Derive the step from simulation time so resets need no extra state.
        step_idx = int(np.round(state["time"][0] / self.step_size))
        rebalance = (step_idx % self.rebalance_every) == 0
        hold_flag = np.full(n, -1.0 if rebalance else 1.0, dtype=np.float32)

        return np.column_stack([lower, upper, hold_flag])
