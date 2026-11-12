"""Small reference policies; no training libraries required."""
import numpy as np


class CashAgent:
    def __init__(self, config):
        self.config = config

    def get_action(self, state):
        return np.tile([0.0, 1.0, 1.0], (self.config.num_trajectories, 1))


class DeployOnceAgent(CashAgent):
    def get_action(self, state):
        n = self.config.num_trajectories
        return np.column_stack([np.full(n, -self.config.tau), np.full(n, self.config.tau),
                                np.where(state["lp_ever_deployed"], 1.0, -1.0)])


class PeriodicAgent(DeployOnceAgent):
    def __init__(self, config):
        super().__init__(config)
        self.step = 0

    def get_action(self, state):
        action = super().get_action(state)
        action[:, 2] = -1.0 if self.step % 100 == 0 else 1.0
        self.step += 1
        return action
