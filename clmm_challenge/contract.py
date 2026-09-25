"""The challenge contract: what a submission must satisfy."""

# --- the class the participant must define -----------------------------------
# The entry-point class name must END WITH this suffix: `Agent`, `MyAgent`,
# `BestAgent`, ... . CLASS_NAME is the privileged exact name: it may omit a base
# class (back-compat) and wins disambiguation when several candidates exist.
CLASS_SUFFIX = "Agent"
CLASS_NAME = "Agent"
INIT_NAME = "__init__"
ACTION_NAME = "get_action"

# `Agent(config)` and `agent.get_action(state)`.
INIT_ARITY = 2
ACTION_ARITY = 2

# Canonical parameter names. The agent runs in a separate process from the
# simulation (lockstep evaluation): __init__ receives a frozen, read-only
# *config* snapshot (plain scalars — no live env, no RNG, no seed), and
# get_action receives the per-step observation *state* dict. `env` is still
# accepted as a deprecated alias for the first __init__ argument so older
# submissions keep validating; the object passed is the config snapshot either
# way (it exposes `num_trajectories`, `tau`, `step_size`, ...).
INIT_PARAM = "config"
INIT_PARAM_LEGACY = "env"
ACTION_PARAM = "state"

# Any class named other than CLASS_NAME must subclass the base Agent, which the
# sandbox exposes as this importable shim module: `from concentrator import Agent`.
# The shim is a minimal local ABC defined in the hosted sandbox — the untrusted process
# never imports SAiFE_gym (the whole simulation stays in the trusted driver).
BASE_AGENT_MODULE = "concentrator"

# --- limits enforced before the code is ever run -----------------------------
MAX_CODE_BYTES = 64 * 1024  # 64 KB of source
