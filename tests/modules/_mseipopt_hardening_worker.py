"""Process-isolated checks for mseipopt memory and signal safety."""

from __future__ import annotations

import gc
import os
import signal
import sys
import warnings
import weakref
from pathlib import Path

import numpy as np

PASS, FAIL, SKIP = 0, 1, 2


def check_memory_boundaries() -> int:
    """Reject unsafe metadata/buffers and retain callbacks through native use."""
    from yapss._private.config import get_conda_prefix
    from yapss._private.mseipopt import bare_np, library

    if get_conda_prefix():
        return SKIP
    library.initialize_ipopt()

    common = {
        "eval_f": lambda x, new_x, out: out.__setitem__((), np.sum((x - 1.0) ** 2)) or True,
        "eval_g": lambda x, new_x, out: True,
        "eval_grad_f": lambda x, new_x, out: out.__setitem__(slice(None), 2 * (x - 1)) or True,
        "eval_jac_g": lambda x, new_x, out: True,
    }
    try:
        bare_np.Problem(
            [-10.0],
            [10.0],
            [],
            [],
            jacobian_structure=([0], [0]),
            **common,
        )
    except ValueError:
        pass
    else:
        print("out-of-range sparse metadata reached native construction")
        return FAIL

    problem = bare_np.Problem(
        [-10.0, -10.0],
        [10.0, 10.0],
        [],
        [],
        jacobian_structure=([], []),
        **common,
    )
    problem.add_int_option("print_level", 0)
    problem.add_str_option("sb", "yes")

    entered = []

    class Intermediate:
        def __call__(self, *args: object) -> bool:
            entered.append(args)
            return True

    callback = Intermediate()
    callback_ref = weakref.ref(callback)
    problem.set_intermediate_callback(callback)
    del callback
    gc.collect()
    if callback_ref() is None:
        print("installed callback was collected before solve")
        return FAIL

    try:
        problem.solve(np.zeros(4)[::2])
    except ValueError:
        pass
    else:
        print("strided solve buffer reached Ipopt")
        return FAIL

    result = problem.solve(np.zeros(2))
    if result.status not in (0, 1) or not entered:
        print(f"positive control failed: status={result.status}, callbacks={len(entered)}")
        return FAIL

    problem.close()
    gc.collect()
    if callback_ref() is not None:
        print("callback remained pinned after native problem close")
        return FAIL
    return PASS


def check_yapss_sigint() -> int:
    """Self-deliver SIGINT during a real YAPSS solve and verify graceful stop."""
    from yapss._private.config import get_conda_prefix
    from yapss.examples.rosenbrock import setup

    if get_conda_prefix():
        return SKIP

    problem = setup()
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.sb = "yes"
    # Keep the user objective on the runtime callback path. Automatic
    # derivatives trace it before Ipopt and would leave nothing to self-signal
    # once YAPSS's temporary handler is installed.
    problem.derivatives.method = "central-difference"
    problem.derivatives.order = "first"
    problem.catch_keyboard_interrupt = True
    original_handler = signal.getsignal(signal.SIGINT)
    original_objective = problem.functions.objective
    original_intermediate = problem._intermediate_cb
    sent = False
    intermediate_results: list[bool] = []

    def objective(arg: object) -> None:
        nonlocal sent
        assert original_objective is not None
        original_objective(arg)
        if not sent and signal.getsignal(signal.SIGINT) != original_handler:
            sent = True
            os.kill(os.getpid(), signal.SIGINT)

    def intermediate(*args: object) -> bool:
        result = original_intermediate(*args)
        intermediate_results.append(result)
        return result

    problem.functions.objective = objective
    type(problem)._intermediate_cb = intermediate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = problem.solve()

    if not sent:
        print("objective never delivered SIGINT during the installed handler")
        return FAIL
    if False not in intermediate_results:
        print(f"intermediate callback never requested stop: {intermediate_results}")
        return FAIL
    if solution.nlp_info.ipopt_status != 5:
        print(f"unexpected Ipopt status: {solution.nlp_info.ipopt_status}")
        return FAIL
    if signal.getsignal(signal.SIGINT) != original_handler:
        print("original SIGINT handler was not restored")
        return FAIL
    if problem._abort or solution.parameter.shape != (2,):
        print("abort state or unconverged solution construction is invalid")
        return FAIL
    return PASS


def check_callback_failures() -> int:
    """Measure each values callback's persistent failure through real Ipopt."""
    from yapss._private.config import get_conda_prefix
    from yapss._private.mseipopt import bare_np, library

    if get_conda_prefix():
        return SKIP
    library.initialize_ipopt()

    callback_names = ("eval_f", "eval_g", "eval_grad_f", "eval_jac_g", "eval_h")

    def run(failing: str | None, use_invalid_point: bool) -> tuple[int, int]:
        calls = 0

        def outcome(name: str) -> bool:
            nonlocal calls
            if name != failing:
                return True
            calls += 1
            if use_invalid_point:
                raise bare_np.InvalidPoint(f"intentional {name} failure")
            return False

        def eval_f(x: np.ndarray, new_x: bool, out: np.ndarray) -> bool:
            out[()] = (x[0] - 1.0) ** 2
            return outcome("eval_f")

        def eval_g(x: np.ndarray, new_x: bool, out: np.ndarray) -> bool:
            out[0] = x[0]
            return outcome("eval_g")

        def eval_grad_f(x: np.ndarray, new_x: bool, out: np.ndarray) -> bool:
            out[0] = 2 * (x[0] - 1.0)
            return outcome("eval_grad_f")

        def eval_jac_g(x: np.ndarray, new_x: bool, out: np.ndarray) -> bool:
            out[0] = 1.0
            return outcome("eval_jac_g")

        def eval_h(
            x: np.ndarray,
            new_x: bool,
            obj_factor: float,
            multipliers: np.ndarray,
            new_multipliers: bool,
            out: np.ndarray,
        ) -> bool:
            out[0] = 2 * obj_factor
            return outcome("eval_h")

        with bare_np.Problem(
            [-10.0],
            [10.0],
            [0.0],
            [0.0],
            eval_f=eval_f,
            eval_g=eval_g,
            eval_grad_f=eval_grad_f,
            jacobian_structure=([0], [0]),
            eval_jac_g=eval_jac_g,
            hessian_structure=([0], [0]),
            eval_h=eval_h,
        ) as problem:
            problem.add_int_option("print_level", 0)
            problem.add_str_option("sb", "yes")
            return problem.solve(np.array([0.5])).status, calls

    positive_status, _ = run(None, False)
    if positive_status not in (0, 1):
        print(f"callback matrix positive control failed: {positive_status}")
        return FAIL

    measured: dict[str, int] = {}
    for use_invalid_point in (False, True):
        label = "InvalidPoint" if use_invalid_point else "false"
        for callback_name in callback_names:
            status, calls = run(callback_name, use_invalid_point)
            measured[f"{label}:{callback_name}"] = status
            if calls == 0 or status != -13:
                print(
                    f"{label} {callback_name} did not fail safely: "
                    f"status={status}, calls={calls}"
                )
                return FAIL
    print(measured)
    return PASS


CHECKS = {
    "callback_failures": check_callback_failures,
    "memory_boundaries": check_memory_boundaries,
    "yapss_sigint": check_yapss_sigint,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        print(f"usage: {Path(__file__).name} <{'|'.join(CHECKS)}>")
        return FAIL
    try:
        return CHECKS[sys.argv[1]]()
    except ImportError as exc:
        print(f"unavailable: {exc}")
        return SKIP


if __name__ == "__main__":
    sys.exit(main())
