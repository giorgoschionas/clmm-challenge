"""Run the real backend evaluator with a fresh nsjail agent for each replay.

Run inside the production worker image with its normal resource/security settings.
Timing wrappers only observe the evaluator; no simulator or sandbox limits change.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, "/app")

import gymnasium
import numpy as np
from app.config import settings
from worker import evaluator, nsjail, sandbox_runner, simulation


def cpu_usage(who):
    value = resource.getrusage(who)
    return value.ru_utime + value.ru_stime


def cgroup_value(name):
    path = Path("/sys/fs/cgroup") / name
    return path.read_text().strip() if path.exists() else None


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode() + b"\0")
        digest.update(str(value.dtype).encode() + str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Standalone periodic/seed-42 profile JSON for exact comparison")
    parser.add_argument("--source-sha256", required=True,
                        help="Expected simulator source digest from the host checkout")
    parser.add_argument("--compare", type=Path, help="Earlier backend benchmark JSON")
    args = parser.parse_args()
    assert settings.sandbox_mode == "nsjail", "The production nsjail sandbox is required"
    assert cgroup_value("memory.max") == "2147483648"
    quota, period = map(int, cgroup_value("cpu.max").split())
    assert quota == 2 * period
    assert cgroup_value("pids.max") == "128"
    binary = nsjail.resolve_binary(settings.nsjail_path)
    assert binary is not None
    root = simulation.resolve_path(settings.clmm_env_path)
    adapter = simulation.load_engine(simulation.CLMM_ENGINE, str(root))
    config = adapter.scenario_config()
    assert (config.n_steps, config.num_trajectories, config.num_ticks) == (10000, 100, 2500)
    reference = json.loads(args.reference.read_text())
    assert reference["seed"] == 42 and reference["policy"] == "periodic"
    assert asdict(config) == reference["scenario"]
    source = hashlib.sha256()
    for path in sorted((root / "clmm_challenge").glob("*.py")):
        source.update(path.name.encode() + b"\0" + path.read_bytes())
    assert source.hexdigest() == args.source_sha256
    backend_source = hashlib.sha256()
    for path in sorted(Path("/app/worker").glob("*.py")) + [Path("/app/app/config.py")]:
        backend_source.update(str(path.relative_to("/app")).encode() + b"\0" + path.read_bytes())
    code = (root / "examples/submission_template.py").read_text()
    passes = []
    original = evaluator._run_lockstep

    def measured_pass(submission_id, env, agent_config, agent_code, cmd, cleanup=(),
                      *, observation_fn, collect_pnl=False):
        assert Path(cmd[0]).resolve() == Path(binary).resolve()
        assert cmd[1] == "--config" and "worker.sandbox_runner" in cmd
        assert (env.n_steps, env.num_trajectories) == (10000, 100)
        jail_config = Path(cmd[2]).read_text()
        config_path = args.output.parent / "nsjail.cfg"
        if config_path.exists():
            assert config_path.read_text() == jail_config
        else:
            config_path.write_text(jail_config)
        observation_count = 0
        pass_number = len(passes) + 1

        def observed_state(state, environment):
            nonlocal observation_count
            observation_count += 1
            if observation_count in (1, 2001, 4001, 6001, 8001):
                print(f"pass {pass_number}/2: {observation_count - 1}/10000 steps completed", flush=True)
            return observation_fn(state, environment)

        started = perf_counter()
        driver_cpu = cpu_usage(resource.RUSAGE_SELF)
        child_cpu = cpu_usage(resource.RUSAGE_CHILDREN)
        result = original(submission_id, env, agent_config, agent_code, cmd, cleanup,
                          observation_fn=observed_state, collect_pnl=collect_pnl)
        record = {
            "pass": pass_number,
            "elapsed_seconds": perf_counter() - started,
            "driver_cpu_seconds": cpu_usage(resource.RUSAGE_SELF) - driver_cpu,
            "child_cpu_seconds": cpu_usage(resource.RUSAGE_CHILDREN) - child_cpu,
            "observations_sent": observation_count,
            "status": result["status"],
            "action_digest": result.get("action_digest"),
            "score": result.get("score"),
            "final_state_sha256": state_digest(env.model_dynamics.state),
            "jail_config_sha256": hashlib.sha256(jail_config.encode()).hexdigest(),
        }
        passes.append(record)
        print(json.dumps(record), flush=True)
        return result

    started_at = datetime.now(timezone.utc).isoformat()
    cgroup_before = cgroup_value("cpu.stat")
    started = perf_counter()
    with patch.object(evaluator, "_run_lockstep", measured_pass):
        result = evaluator.evaluate(
            "nsjail-public-benchmark", code, simulation_env_path=str(root),
            seeds=[42], engine_id=simulation.CLMM_ENGINE, verify_reproducibility=True,
        )
    elapsed = perf_counter() - started
    matched = (result.get("status") == "scored"
               and result.get("reproducibility_verified") is True
               and len(passes) == 2
               and all(p["observations_sent"] == 10000 for p in passes)
               and all(p["final_state_sha256"] == reference["final_state_sha256"] for p in passes)
               and result["score"] == reference["score"]
               and result["seed_results"][0]["returns"] == reference["returns"]
               and result["seed_results"][0]["action_digests"] == [reference["action_sha256"]] * 2)
    backend_match = None
    if args.compare:
        previous = json.loads(args.compare.read_text())
        backend_match = all(result.get(key) == previous["result"].get(key)
                            for key in ("status", "score", "pnl_pct", "seed_results",
                                        "reproducibility_verified"))
        assert backend_match, "Backend outcomes differ from reference"
        assert previous["runtime"] == {
            "python": platform.python_version(), "numpy": np.__version__,
            "gymnasium": gymnasium.__version__, "platform": platform.platform(),
            "machine": platform.machine()}
        assert previous["backend_source_sha256"] == backend_source.hexdigest()
        assert previous["nsjail_binary_sha256"] == hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    report = {
        "exact_match_to_backend_reference": backend_match,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at_utc": started_at,
        "sandbox_mode": settings.sandbox_mode,
        "seed": 42,
        "policy": "examples/submission_template.py",
        "scenario": asdict(config),
        "elapsed_seconds": elapsed,
        "passes": passes,
        "reproducibility_verified": result.get("reproducibility_verified", False),
        "exact_match_to_original_reference": matched,
        "simulator_source_sha256": source.hexdigest(),
        "backend_source_sha256": backend_source.hexdigest(),
        "agent_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "nsjail_binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "gymnasium": gymnasium.__version__, "platform": platform.platform(),
                    "machine": platform.machine()},
        "packages": dict(sorted((package.metadata["Name"], package.version) for package in distributions())),
        "limits": {"memory_max": cgroup_value("memory.max"),
                   "cpu_max": cgroup_value("cpu.max"), "pids_max": cgroup_value("pids.max"),
                   "agent_cpu_seconds": sandbox_runner.CPU_SECONDS,
                   "agent_memory_bytes": sandbox_runner.MEMORY_BYTES,
                   "pass_wall_seconds": evaluator.WALL_CLOCK_TIMEOUT,
                   "per_action_seconds": evaluator.STEP_TIMEOUT},
        "resources": {"container_peak_memory_bytes": cgroup_value("memory.peak"),
                      "cpu_stat_before": cgroup_before, "cpu_stat_after": cgroup_value("cpu.stat"),
                      "driver_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss},
        "scope": "Complete evaluator.evaluate call, two fresh nsjail agents, framed IPC, scoring and replay comparison; excludes image build, container/Python startup, HTTP, queue, and database.",
        "result": result,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"elapsed_seconds": elapsed, "status": result["status"],
                      "reproducibility_verified": report["reproducibility_verified"],
                      "exact_match_to_original_reference": matched}), flush=True)
    assert matched, result.get("message", "reference comparison failed")


if __name__ == "__main__":
    main()
