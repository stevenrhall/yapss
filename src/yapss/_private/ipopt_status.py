"""

The status Ipopt reports at the end of a solve.

`IpoptStatus` names Ipopt's ``ApplicationReturnStatus`` codes (``IpReturnCodes_inc.h``) and
says, for each, what Ipopt printed and whether the solve converged. `status_or_raise` turns
the integer the C interface returns into an `IpoptStatus`, or raises when Ipopt stopped
without an iterate to report.

Which statuses carry an iterate is read from Ipopt 3.14.11, ``IpoptApplication::
call_optimize``: ``FinalizeSolution`` receives the current iterate -- its constraint values,
objective, and multipliers -- only for the statuses in `_WITH_ITERATE`. For every other
status Ipopt either never calls it, leaving the output arrays as they were passed in, or
calls it with the constraints and every multiplier set to zero. A `Solution` built from
those arrays would look like a solution and be nothing of the kind.
"""

# future imports
from __future__ import annotations

# standard imports
from enum import IntEnum

__all__ = ["IpoptStatus", "status_or_raise"]


class IpoptStatus(IntEnum):
    """The status Ipopt reports at the end of a solve.

    An `IntEnum`, so it compares equal to Ipopt's integer code: ``solution.status == 0``
    and ``solution.status == IpoptStatus.SOLVE_SUCCEEDED`` are the same test.
    """

    SOLVE_SUCCEEDED = 0
    SOLVED_TO_ACCEPTABLE_LEVEL = 1
    INFEASIBLE_PROBLEM_DETECTED = 2
    SEARCH_DIRECTION_BECOMES_TOO_SMALL = 3
    DIVERGING_ITERATES = 4
    USER_REQUESTED_STOP = 5
    FEASIBLE_POINT_FOUND = 6
    MAXIMUM_ITERATIONS_EXCEEDED = -1
    RESTORATION_FAILED = -2
    ERROR_IN_STEP_COMPUTATION = -3
    MAXIMUM_CPUTIME_EXCEEDED = -4
    MAXIMUM_WALLTIME_EXCEEDED = -5
    NOT_ENOUGH_DEGREES_OF_FREEDOM = -10
    INVALID_PROBLEM_DEFINITION = -11
    INVALID_OPTION = -12
    INVALID_NUMBER_DETECTED = -13
    UNRECOVERABLE_EXCEPTION = -100
    NONIPOPT_EXCEPTION_THROWN = -101
    INSUFFICIENT_MEMORY = -102
    INTERNAL_ERROR = -199

    @property
    def message(self) -> str:
        """Ipopt's own description of the status, as its ``EXIT:`` line prints it."""
        return _MESSAGES[self]

    @property
    def converged(self) -> bool:
        """Whether the solve converged: statuses 0, 1, and 6.

        1 ("Solved To Acceptable Level") counts. It is the normal outcome when tolerances
        are pushed hard, and the answer is routinely correct to more digits than requested.
        6 is the converged outcome of a square problem, which has no objective to optimize.
        """
        return self in _CONVERGED


# Transcribed from the `EXIT:` lines Ipopt prints in `IpIpoptApplication.cpp::call_optimize`,
# verbatim and with their punctuation, so that the message YAPSS reports and the console
# line directly above it are the same string. Status -101 is the exception: that branch
# reports the exception and prints no `EXIT:` line, so the text is YAPSS's own.
_MESSAGES = {
    IpoptStatus.SOLVE_SUCCEEDED: "Optimal Solution Found.",
    IpoptStatus.SOLVED_TO_ACCEPTABLE_LEVEL: "Solved To Acceptable Level.",
    IpoptStatus.INFEASIBLE_PROBLEM_DETECTED: (
        "Converged to a point of local infeasibility. Problem may be infeasible."
    ),
    IpoptStatus.SEARCH_DIRECTION_BECOMES_TOO_SMALL: "Search Direction is becoming Too Small.",
    IpoptStatus.DIVERGING_ITERATES: "Iterates diverging; problem might be unbounded.",
    IpoptStatus.USER_REQUESTED_STOP: (
        "Stopping optimization at current point as requested by user."
    ),
    IpoptStatus.FEASIBLE_POINT_FOUND: "Feasible point for square problem found.",
    IpoptStatus.MAXIMUM_ITERATIONS_EXCEEDED: "Maximum Number of Iterations Exceeded.",
    IpoptStatus.RESTORATION_FAILED: "Restoration Failed!",
    IpoptStatus.ERROR_IN_STEP_COMPUTATION: "Error in step computation!",
    IpoptStatus.MAXIMUM_CPUTIME_EXCEEDED: "Maximum CPU time exceeded.",
    IpoptStatus.MAXIMUM_WALLTIME_EXCEEDED: "Maximum wallclock time exceeded.",
    IpoptStatus.NOT_ENOUGH_DEGREES_OF_FREEDOM: "Problem has too few degrees of freedom.",
    IpoptStatus.INVALID_PROBLEM_DEFINITION: (
        "Problem has inconsistent variable bounds or constraint sides."
    ),
    IpoptStatus.INVALID_OPTION: "Invalid option encountered.",
    IpoptStatus.INVALID_NUMBER_DETECTED: "Invalid number in NLP function or derivative detected.",
    IpoptStatus.UNRECOVERABLE_EXCEPTION: "Some uncaught Ipopt exception encountered.",
    IpoptStatus.NONIPOPT_EXCEPTION_THROWN: (
        "An exception not raised by Ipopt was caught during the solve."
    ),
    IpoptStatus.INSUFFICIENT_MEMORY: "Not enough memory.",
    IpoptStatus.INTERNAL_ERROR: (
        "INTERNAL ERROR: Unknown SolverReturn value - Notify IPOPT Authors."
    ),
}

_CONVERGED = frozenset(
    {
        IpoptStatus.SOLVE_SUCCEEDED,
        IpoptStatus.SOLVED_TO_ACCEPTABLE_LEVEL,
        IpoptStatus.FEASIBLE_POINT_FOUND,
    },
)

# The statuses for which Ipopt reports its current iterate (see the module docstring).
_WITH_ITERATE = frozenset(
    {
        IpoptStatus.SOLVE_SUCCEEDED,
        IpoptStatus.SOLVED_TO_ACCEPTABLE_LEVEL,
        IpoptStatus.INFEASIBLE_PROBLEM_DETECTED,
        IpoptStatus.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
        IpoptStatus.DIVERGING_ITERATES,
        IpoptStatus.USER_REQUESTED_STOP,
        IpoptStatus.FEASIBLE_POINT_FOUND,
        IpoptStatus.MAXIMUM_ITERATIONS_EXCEEDED,
        IpoptStatus.RESTORATION_FAILED,
        IpoptStatus.ERROR_IN_STEP_COMPUTATION,
        IpoptStatus.MAXIMUM_CPUTIME_EXCEEDED,
        IpoptStatus.MAXIMUM_WALLTIME_EXCEEDED,
    },
)

# What to tell the user for each status without an iterate, after Ipopt's own message.
_WITHOUT_ITERATE: dict[IpoptStatus, tuple[type[Exception], str]] = {
    IpoptStatus.NOT_ENOUGH_DEGREES_OF_FREEDOM: (
        ValueError,
        (
            "The problem has more equality constraints than free decision variables, so "
            "there is nothing left to optimize. A variable whose lower and upper bounds are "
            "equal is fixed, not free, and a constraint whose lower and upper bounds are "
            "equal is an equality."
        ),
    ),
    IpoptStatus.INVALID_PROBLEM_DEFINITION: (
        ValueError,
        "A lower bound is above its upper bound for a variable or constraint.",
    ),
    IpoptStatus.INVALID_OPTION: (
        ValueError,
        "Ipopt refused an option setting; its output above names the option.",
    ),
    IpoptStatus.INVALID_NUMBER_DETECTED: (
        ValueError,
        (
            "A callback or one of its derivatives returned NaN or Inf during the solve, so "
            "Ipopt has no constraint values or multipliers to report. Common causes are a "
            "square root or logarithm of a value that became negative or zero, a division by "
            "a value that became zero, and a power of a negative base."
        ),
    ),
    IpoptStatus.UNRECOVERABLE_EXCEPTION: (
        RuntimeError,
        "This is a failure inside Ipopt rather than a problem with the problem definition.",
    ),
    IpoptStatus.NONIPOPT_EXCEPTION_THROWN: (
        RuntimeError,
        "This is a failure inside Ipopt rather than a problem with the problem definition.",
    ),
    IpoptStatus.INSUFFICIENT_MEMORY: (MemoryError, ""),
    IpoptStatus.INTERNAL_ERROR: (
        RuntimeError,
        "This is a failure inside Ipopt rather than a problem with the problem definition.",
    ),
}


def status_or_raise(code: int) -> IpoptStatus:
    """Return Ipopt's status code as an `IpoptStatus`, raising if there is no iterate.

    Parameters
    ----------
    code : int
        The status the Ipopt C interface returned.

    Raises
    ------
    ValueError
        For a status caused by the problem or its callbacks: too few degrees of freedom,
        inconsistent bounds, an invalid option, or a NaN or Inf from a callback.
    MemoryError
        When Ipopt ran out of memory.
    RuntimeError
        For a failure inside Ipopt, or a status this version of YAPSS does not know.
    """
    try:
        status = IpoptStatus(code)
    except ValueError:
        msg = (
            f"Ipopt returned status {code}, which this version of YAPSS does not recognize, "
            f"so it cannot tell whether the result is a solution."
        )
        raise RuntimeError(msg) from None
    if status in _WITH_ITERATE:
        return status
    exception, explanation = _WITHOUT_ITERATE[status]
    msg = f'Ipopt stopped without a solution. Status {int(status)}: "{status.message}"'
    if explanation:
        msg += f"\n{explanation}"
    raise exception(msg)
