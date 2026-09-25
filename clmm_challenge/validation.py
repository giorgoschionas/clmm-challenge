"""Submission validation.

structural_check() is static and safe. It runs in the trusted worker/API before
any code executes (``ast.parse`` does not run the code), rejecting junk fast
without spending a sandbox on it.
"""
from __future__ import annotations

import ast

from . import contract


def _positional_arity(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return len(fn.args.posonlyargs) + len(fn.args.args)


def _method_named(cls: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _select_entry_class(tree: ast.Module) -> tuple[ast.ClassDef | None, str]:
    """Pick the submission's entry-point class from the top-level definitions.

    The entry point is the class whose name ends with ``Agent``. A class named
    exactly ``Agent`` wins (and is the only one allowed to omit a base class);
    otherwise exactly one ``...Agent`` class must exist. This mirrors the runtime
    discovery in ``sandbox_runner`` so the two layers never disagree.
    """
    candidates = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith(contract.CLASS_SUFFIX)
    ]
    if not candidates:
        return None, (
            f"must define a top-level class whose name ends with "
            f"'{contract.CLASS_SUFFIX}' (e.g. class MyAgent(Agent))"
        )

    exact = [c for c in candidates if c.name == contract.CLASS_NAME]
    if exact:
        return exact[-1], ""  # a later definition shadows an earlier one at runtime
    if len(candidates) == 1:
        return candidates[0], ""

    names = ", ".join(sorted(c.name for c in candidates))
    return None, (
        f"multiple candidate Agent classes ({names}); name your entry point "
        f"'{contract.CLASS_NAME}' or rename the others so only one class name "
        f"ends with '{contract.CLASS_SUFFIX}'"
    )


def structural_check(code: str) -> tuple[bool, str]:
    """Cheap, safe pre-checks. Returns (ok, message)."""
    if len(code.encode("utf-8")) > contract.MAX_CODE_BYTES:
        return False, f"submission exceeds {contract.MAX_CODE_BYTES} bytes"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg} (line {e.lineno})"

    cls, msg = _select_entry_class(tree)
    if cls is None:
        return False, msg

    # A renamed entry point must declare a base class. Verifying it is *really*
    # the base Agent needs the live class object, so the authoritative check is
    # issubclass() at runtime; here we only reject the obvious "no base at all".
    if cls.name != contract.CLASS_NAME and not cls.bases:
        return False, (
            f"'{cls.name}' must subclass the base Agent "
            f"(from {contract.BASE_AGENT_MODULE} import Agent); only a class named "
            f"'{contract.CLASS_NAME}' may omit it"
        )

    init = _method_named(cls, contract.INIT_NAME)
    if init is None:
        return False, f"'{cls.name}' must define {contract.INIT_NAME}(self, {contract.INIT_PARAM})"
    if isinstance(init, ast.AsyncFunctionDef):
        return False, f"'{cls.name}.{contract.INIT_NAME}' must be a regular method"
    if _positional_arity(init) != contract.INIT_ARITY:
        return False, (
            f"'{cls.name}.{contract.INIT_NAME}' must take self and {contract.INIT_PARAM}"
        )
    init_args = init.args.posonlyargs + init.args.args
    if init_args[0].arg != "self" or init_args[1].arg not in (
        contract.INIT_PARAM, contract.INIT_PARAM_LEGACY
    ):
        return False, (
            f"'{cls.name}.{contract.INIT_NAME}' must be __init__(self, {contract.INIT_PARAM})"
        )

    get_action = _method_named(cls, contract.ACTION_NAME)
    if get_action is None:
        return False, f"'{cls.name}' must define {contract.ACTION_NAME}(self, {contract.ACTION_PARAM})"
    if isinstance(get_action, ast.AsyncFunctionDef):
        return False, f"'{cls.name}.{contract.ACTION_NAME}' must be a regular method"
    if _positional_arity(get_action) != contract.ACTION_ARITY:
        return False, (
            f"'{cls.name}.{contract.ACTION_NAME}' must take self and {contract.ACTION_PARAM}"
        )
    action_args = get_action.args.posonlyargs + get_action.args.args
    if action_args[0].arg != "self" or action_args[1].arg != contract.ACTION_PARAM:
        return False, (
            f"'{cls.name}.{contract.ACTION_NAME}' must be "
            f"{contract.ACTION_NAME}(self, {contract.ACTION_PARAM})"
        )

    return True, "ok"
