"""Local practice for trusted strategy files. Hosted grading uses the backend sandbox."""
import abc
import argparse
import ast
import csv
from pathlib import Path
import random
import sys
import types

import numpy as np

from .baselines import CashAgent, DeployOnceAgent, PeriodicAgent
from .challenge import PRACTICE_SEEDS, ScenarioConfig, create_environment, official_observation, validate_action
from .contract import MAX_CODE_BYTES
from .validation import structural_check, _select_entry_class


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state):
        pass

    def get_expected_action(self, state, n_samples=1000):
        return np.array([self.get_action(state) for _ in range(n_samples)]).mean(axis=0)


def load_agent(path: Path, policy_seed: int = 0):
    if path.stat().st_size > MAX_CODE_BYTES:
        raise ValueError("submission exceeds 65536 bytes")
    source = path.read_text(encoding="utf-8")
    ok, message = structural_check(source)
    if not ok:
        raise ValueError(message)
    # Compatibility with the existing hosted Agent shim. Local code is trusted:
    # this loader is deliberately not advertised as a security boundary.
    shim = types.ModuleType("concentrator")
    shim.Agent = BaseAgent
    sys.modules["concentrator"] = shim
    namespace = {"__name__": "submission", "__file__": str(path)}
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    exec(compile(source, str(path), "exec"), namespace)
    node, _ = _select_entry_class(ast.parse(source))
    agent = namespace[node.name]
    if not isinstance(agent, type) or (node.name != "Agent" and not issubclass(agent, BaseAgent)):
        raise ValueError("a renamed Agent must subclass concentrator.Agent")
    agent._practice_source_path = path
    return agent


def run_episode(env, agent):
    state, _ = env.reset()
    initial = state["portfolio_value"].copy()
    returns = np.zeros(env.num_trajectories)
    for _ in range(env.n_steps):
        action = validate_action(agent.get_action(official_observation(state, env)), env.num_trajectories)
        state, reward, terminated, truncated, _ = env.step(action)
        returns += reward
        if np.all(terminated | truncated):
            break
    pnl = state["portfolio_value"] - initial
    if not np.isfinite(returns).all() or not np.isfinite(pnl).all():
        raise ValueError("simulation produced non-finite results")
    return returns, pnl


def evaluate_agent(agent_class, config, seeds):
    scores, pnls = [], []
    for seed in seeds:
        random.seed(config.policy_seed)
        np.random.seed(config.policy_seed)
        source_path = getattr(agent_class, "_practice_source_path", None)
        current_class = load_agent(source_path, config.policy_seed) if source_path else agent_class
        env = create_environment(config, seed)
        score, pnl = run_episode(env, current_class(config.submission_namespace()))
        scores.append(score)
        pnls.append(pnl)
    scores, pnls = np.concatenate(scores), np.concatenate(pnls)
    return {"score": float(scores.mean()), "mean_pnl": float(pnls.mean()),
            "pnl_pct": float(100 * pnls.mean() / config.initial_wealth),
            "mean_inventory_penalty": float((pnls - scores).mean()),
            "num_paths": len(scores)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--agent", type=Path)
    source.add_argument("--submissions-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PRACTICE_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("results/leaderboard.csv"))
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args(argv)
    paths = [args.agent] if args.agent else sorted(args.submissions_dir.glob("*.py"))
    if not paths:
        parser.error("no Python submissions found")
    config, rows, failed = ScenarioConfig(), [], False
    for path in paths:
        try:
            rows.append({"name": path.stem, "kind": "submission",
                         **evaluate_agent(load_agent(path), config, args.seeds)})
        except Exception as exc:
            failed = True
            print(f"{path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not args.no_baselines:
        for agent in (CashAgent, DeployOnceAgent, PeriodicAgent):
            rows.append({"name": agent.__name__, "kind": "baseline",
                         **evaluate_agent(agent, config, args.seeds)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    print("Rank | Agent | Inventory-adjusted score | Raw PnL | Inventory penalty")
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        print(f"{rank} | {row['name']} | {row['score']:+.4f} | {row['mean_pnl']:+.4f} | {row['mean_inventory_penalty']:.4f}")
    if rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Practice results: {args.output}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
