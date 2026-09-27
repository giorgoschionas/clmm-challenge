from pathlib import Path

from clmm_challenge.challenge import ScenarioConfig
from clmm_challenge.evaluate import evaluate_agent, load_agent


def test_module_state_and_seeded_randomness_reset_between_seeds(tmp_path):
    path = tmp_path / "policy.py"
    path.write_text('''import random
import numpy as np
width = random.randint(1, 50)
calls = 0
class Agent:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.policy_seed)
    def get_action(self, state):
        global calls
        calls += 1
        assert calls <= self.config.n_steps
        w = min(width, int(self.rng.integers(1, 51)))
        return np.tile([-w, w, -1], (self.config.num_trajectories, 1))
''')
    config = ScenarioConfig(num_trajectories=3, n_steps=8, terminal_time=0.008, num_ticks=200)
    agent = load_agent(path)
    first = evaluate_agent(agent, config, [1009, 2027])
    assert first == evaluate_agent(agent, config, [1009, 2027])
    assert first["num_paths"] == 6
