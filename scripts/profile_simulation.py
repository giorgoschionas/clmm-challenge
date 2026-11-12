"""Profile a public practice episode without changing the official scenario.

Use --root to benchmark an archived checkout with the same interpreter and harness.
Optional traces hash every state array, reward, action, arrival and diagnostic at
every step; they add overhead and should be used separately from timing runs.
"""
import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import pstats
import sys
from time import perf_counter, process_time


def digest_arrays(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode() + b"\0")
        digest.update(str(value.dtype).encode() + str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=("periodic", "cash", "deploy_once", "mixed"), default="periodic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compare", type=Path, help="Require exact results matching an earlier JSON report")
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    import numpy as np
    from clmm_challenge.challenge import ScenarioConfig, create_environment, official_observation
    from examples.baselines import CashAgent, DeployOnceAgent, PeriodicAgent
    import clmm_challenge

    assert Path(clmm_challenge.__file__).resolve().is_relative_to(args.root.resolve())

    config = ScenarioConfig()
    env = create_environment(config, args.seed)
    agent = {"cash": CashAgent, "deploy_once": DeployOnceAgent,
             "periodic": PeriodicAgent, "mixed": PeriodicAgent}[args.policy](config.submission_namespace())
    state, _ = env.reset()
    initial_value = state["portfolio_value"].copy()
    returns = np.zeros(config.num_trajectories)
    actions = hashlib.sha256()
    trace = [digest_arrays(state)] if args.trace else None
    # Vary widths, asymmetric ranges, and hold/rebalance timing across paths.
    # This deterministic fixture never consumes the market RNG.
    policy_rng = np.random.default_rng(20261112)
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    started, cpu_started = perf_counter(), process_time()
    for step in range(config.n_steps):
        if args.policy == "mixed":
            action = np.column_stack((policy_rng.integers(-50, 0, config.num_trajectories),
                                      policy_rng.integers(1, 51, config.num_trajectories),
                                      np.where((step + np.arange(config.num_trajectories)) % 137 == 0, -1.0, 1.0)))
        else:
            action = agent.get_action(official_observation(state, env))
        actions.update(action.astype("<f8", copy=False).tobytes())
        state, reward, terminated, truncated, info = env.step(action)
        returns += reward
        if trace is not None:
            trace.append(digest_arrays({**state, **{"info/" + k: v for k, v in info.items()},
                                       "reward": reward, "action": action, "terminated": terminated,
                                       "truncated": truncated, "arrivals": env.model_dynamics.last_arrivals}))
        if (step + 1) % 2000 == 0:
            print(f"{args.policy} seed {args.seed}: {step + 1}/{config.n_steps} steps", flush=True)
    elapsed, cpu = perf_counter() - started, process_time() - cpu_started
    if profiler:
        profiler.disable()
    assert terminated.all() and not truncated.any()
    assert np.isfinite(returns).all()
    source = hashlib.sha256()
    for path in sorted((args.root / "clmm_challenge").glob("*.py")):
        source.update(path.name.encode() + b"\0" + path.read_bytes())
    report = {"scenario": asdict(config), "seed": args.seed, "policy": args.policy,
              "python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
              "source_sha256": source.hexdigest(), "elapsed_seconds": elapsed, "cpu_seconds": cpu,
              "profiled": args.profile, "traced": args.trace, "steps_completed": step + 1,
              "score": float(returns.mean()), "returns": returns.tolist(),
              "pnl": (state["portfolio_value"] - initial_value).tolist(),
              "action_sha256": actions.hexdigest(), "final_state_sha256": digest_arrays(state)}
    if trace is not None:
        report["step_sha256"] = trace
    if args.compare:
        reference = json.loads(args.compare.read_text())
        keys = ["scenario", "seed", "policy", "steps_completed", "score", "returns", "pnl",
                "action_sha256", "final_state_sha256"]
        if "step_sha256" in reference:
            keys.append("step_sha256")
        for key in keys:
            if report.get(key) != reference[key]:
                raise AssertionError(f"Reference mismatch: {key}")
        report["comparison"] = {"exact_match": True, "reference_source_sha256": reference["source_sha256"],
                                "checked_fields": keys, "step_states_checked": len(trace) if trace else 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if profiler:
        profiler.dump_stats(str(args.output.with_suffix(".prof")))
        stats = pstats.Stats(profiler)
        report["profile"] = sorted(
            ({"file": Path(file).name, "line": line, "function": name,
              "calls": nc, "self_seconds": tt, "cumulative_seconds": ct}
             for (file, line, name), (_, nc, tt, ct, _) in stats.stats.items()),
            key=lambda row: row["cumulative_seconds"], reverse=True)[:50]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("elapsed_seconds", "cpu_seconds", "score", "final_state_sha256")}))


if __name__ == "__main__":
    main()
