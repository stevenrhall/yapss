"""Shared helpers for the contract suite.

The contract suite states the user contract of YAPSS as executable clauses: what a user
may do (every accepted form of input, stored and used as documented) and what a user
may get wrong (each mistake raises or warns with the promised type, message, and
location). Each test is one clause.

A clause the code does not meet yet is marked `not_yet(...)`, a strict xfail naming the
decision or work-list item that will meet it. When that change lands, the test passes,
strict xfail turns the pass into a failure, and the marker must be removed in the same
commit. `pytest tests/contract -rx` lists everything outstanding.

A clause marked `proposed(...)` is also a strict xfail, but its behavior is not yet
decided. Decide it, then either keep the test as `not_yet` or delete it.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from yapss import Problem

# Set YAPSS_ERROR_CATALOG to a path and every message `raises` sees is appended there, one
# JSON object per line, for `tools/error_catalog.py` to render. Off by default, so an
# ordinary run is untouched; one line per record, appended, because xdist runs the suite in
# several processes at once.
_CATALOG = os.environ.get("YAPSS_ERROR_CATALOG")


def _record(exc: BaseException, statement: str, test: str, area: str) -> None:
    """Append one raised message to the catalogue."""
    entry = {
        "area": area,
        "type": type(exc).__name__,
        "message": "\n".join([str(exc), *getattr(exc, "__notes__", [])]),
        "statement": statement.strip(),
        "test": test,
    }
    with Path(_CATALOG).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


def problem() -> Problem:
    """Return a two-phase problem with at least one of every kind of variable.

    Phase 0 has two states, so slices and multi-element arrays are exercised; phase 1
    has one of everything, so single-element arrays are too.
    """
    return Problem(name="contract", nx=[2, 1], nu=[1, 1], nq=[1, 1], nh=[1, 1], ns=2, nd=2)


def not_yet(item: str, clause: str) -> pytest.MarkDecorator:
    """Mark a clause the code does not meet yet, naming the item that will meet it."""
    return pytest.mark.xfail(strict=True, reason=f"{item}: {clause}")


def proposed(clause: str) -> pytest.MarkDecorator:
    """Mark a clause whose behavior is proposed but not yet decided."""
    return pytest.mark.xfail(strict=True, reason=f"PROPOSED, not decided: {clause}")


@contextmanager
def raises(exc: type[BaseException], *fragments: str, at: str | None = None) -> Iterator[Any]:
    """Assert that the block raises `exc` with every fragment in its message or its notes.

    Notes count as message: Python prints them directly under the message, which is where a
    user reads them (YAPSS adds one naming the callback an exception came from).

    With `at`, also assert that the exception was raised by the statement in the block
    whose source contains `at` -- that is, the traceback's last frame in the calling test
    file is that statement, the stand-in for the user's own line.
    """
    caller = sys._getframe(2)  # 0: here, 1: contextmanager, 2: test
    caller_file = caller.f_code.co_filename
    with pytest.raises(exc) as info:
        yield info
    message = "\n".join([str(info.value), *getattr(info.value, "__notes__", [])])
    missing = [fragment for fragment in fragments if fragment not in message]
    assert not missing, f"message {message!r} lacks {missing!r}"
    frames = [
        frame
        for frame in traceback.extract_tb(info.value.__traceback__)
        if frame.filename == caller_file
    ]
    if at is not None:
        assert frames, "the traceback does not pass through the calling test"
        line = frames[-1].line or ""
        assert at in line, f"raised by the statement {line!r}, not by the one containing {at!r}"
    if _CATALOG:
        _record(
            info.value,
            frames[-1].line or "" if frames else "",
            caller.f_code.co_name,
            Path(caller_file).stem.replace("test_contract_", ""),
        )


def assert_float64_array(value: Any, expected: Any) -> None:
    """Assert that a stored value is a float64 ndarray equal to `expected`."""
    assert isinstance(value, np.ndarray), type(value)
    assert value.dtype == np.float64, value.dtype
    np.testing.assert_array_equal(value, np.asarray(expected, dtype=np.float64))


# Forms in which a user may supply a 1-D sequence of n real numbers. Each maps n to a
# value whose float64 conversion is `reference(n)`.
def reference(n: int) -> list[float]:
    """Return the reference values the 1-D forms below encode."""
    return [float(i + 1) for i in range(n)]


SEQUENCE_FORMS = {
    "list of int": lambda n: [i + 1 for i in range(n)],
    "list of float": lambda n: [float(i + 1) for i in range(n)],
    "tuple of float": lambda n: tuple(float(i + 1) for i in range(n)),
    "range": lambda n: range(1, n + 1),
    "ndarray int64": lambda n: np.arange(1, n + 1, dtype=np.int64),
    "ndarray float32": lambda n: np.arange(1, n + 1, dtype=np.float32),
    "ndarray float64": lambda n: np.arange(1, n + 1, dtype=np.float64),
    "list of numpy scalars": lambda n: [
        np.float32(i + 1) if i % 2 else np.int64(i + 1) for i in range(n)
    ],
}

# Forms in which a user may supply one real number.
SCALAR_FORMS = {
    "int": 3,
    "float": 3.0,
    "numpy int64": np.int64(3),
    "numpy float32": np.float32(3.0),
    "numpy float64": np.float64(3.0),
}


# A small solvable problem for callback and solve contracts: the brachistochrone with a
# path constraint on speed, an integral of the control squared, and a discrete constraint
# on the final height, so every output kind is exercised. Three segments of four points
# keep each solve to milliseconds.
G0 = 32.174
METHODS = ("auto", "central-difference", "central-difference-full")


def default_continuous(arg: Any) -> None:
    """Dynamics, one path constraint, and one integrand for every phase in phase_list."""
    import yapss.math as ym

    for p in arg.phase_list:
        _, _, v = arg.phase[p].state
        (u,) = arg.phase[p].control
        arg.phase[p].dynamics[:] = v * ym.cos(u), v * ym.sin(u), G0 * ym.sin(u)
        arg.phase[p].path[:] = (v,)
        arg.phase[p].integrand[:] = (u**2,)


def default_objective(arg: Any) -> None:
    """Minimize final time, with a small penalty on the integral."""
    arg.objective = arg.phase[0].final_time + 1e-3 * arg.phase[0].integral[0]


def default_discrete(arg: Any) -> None:
    """Constrain the final height."""
    arg.discrete[:] = (arg.phase[0].final_state[1],)


def callback_problem(
    method: str = "auto",
    *,
    continuous: Any = default_continuous,
    objective: Any = default_objective,
    discrete: Any = default_discrete,
) -> Problem:
    """Return the small callback-contract problem with the given callbacks and method."""
    ocp = Problem(name="callbacks", nx=[3], nu=[1], nq=[1], nh=[1], nd=1)
    ocp.functions.objective = objective
    ocp.functions.continuous = continuous
    ocp.functions.discrete = discrete
    b = ocp.bounds.phase[0]
    b.initial_time.lower = b.initial_time.upper = 0.0
    b.initial_state.lower[:] = b.initial_state.upper[:] = 0.0
    b.final_state.lower[0] = b.final_state.upper[0] = 1.0
    b.state.lower[:] = 0.0
    b.state.upper[:] = 10.0
    b.control.lower[:] = -np.pi / 2
    b.control.upper[:] = np.pi / 2
    b.path.lower[:] = 0.0
    b.path.upper[:] = 100.0
    ocp.bounds.discrete.lower[:] = ocp.bounds.discrete.upper[:] = 0.5
    g = ocp.guess.phase[0]
    g.time = [0.0, 1.0]
    g.state = [[0.0, 1.0], [0.0, 0.5], [0.0, 5.0]]
    g.control = [[0.0, 0.0]]
    ocp.mesh.phase[0].collocation_points = (4, 4, 4)
    ocp.mesh.phase[0].fraction = (1 / 3, 1 / 3, 1 / 3)
    ocp.derivatives.method = method
    ocp.ipopt_options.print_level = 0
    return ocp
