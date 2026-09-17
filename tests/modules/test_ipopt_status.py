"""Tests for `yapss._private.ipopt_status`."""

import re

import pytest

import yapss
from yapss import IpoptStatus
from yapss._private.ipopt_status import status_or_raise
from yapss._private.solution import QUIET_IPOPT_STATUSES

# `ApplicationReturnStatus` in Ipopt 3.14.11's IpReturnCodes_inc.h, copied by hand: the enum
# must name every code Ipopt can return, with Ipopt's own spelling in upper case.
IPOPT_RETURN_CODES = {
    "Solve_Succeeded": 0,
    "Solved_To_Acceptable_Level": 1,
    "Infeasible_Problem_Detected": 2,
    "Search_Direction_Becomes_Too_Small": 3,
    "Diverging_Iterates": 4,
    "User_Requested_Stop": 5,
    "Feasible_Point_Found": 6,
    "Maximum_Iterations_Exceeded": -1,
    "Restoration_Failed": -2,
    "Error_In_Step_Computation": -3,
    "Maximum_CpuTime_Exceeded": -4,
    "Maximum_WallTime_Exceeded": -5,
    "Not_Enough_Degrees_Of_Freedom": -10,
    "Invalid_Problem_Definition": -11,
    "Invalid_Option": -12,
    "Invalid_Number_Detected": -13,
    "Unrecoverable_Exception": -100,
    "NonIpopt_Exception_Thrown": -101,
    "Insufficient_Memory": -102,
    "Internal_Error": -199,
}

# Statuses for which Ipopt 3.14.11's call_optimize passes its iterate to FinalizeSolution.
WITH_ITERATE = [0, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5]

WITHOUT_ITERATE = {
    -10: ValueError,
    -11: ValueError,
    -12: ValueError,
    -13: ValueError,
    -100: RuntimeError,
    -101: RuntimeError,
    -102: MemoryError,
    -199: RuntimeError,
}


def test_members_are_ipopts_return_codes():
    assert {member.name: member.value for member in IpoptStatus} == {
        name.upper(): code for name, code in IPOPT_RETURN_CODES.items()
    }


def test_status_is_public_and_an_int():
    assert "IpoptStatus" in yapss.__all__
    assert IpoptStatus.SOLVE_SUCCEEDED == 0
    assert isinstance(IpoptStatus(-1), int)
    assert f"{IpoptStatus(-1)}" == "-1"  # messages that format a status show the code


@pytest.mark.parametrize("status", list(IpoptStatus))
def test_every_status_has_a_message(status):
    assert status.message
    assert status.message.endswith((".", "!"))


@pytest.mark.parametrize("status", list(IpoptStatus))
def test_converged_is_statuses_0_1_and_6(status):
    assert status.converged is (status in (0, 1, 6))


def test_quiet_statuses_are_the_converged_ones():
    assert QUIET_IPOPT_STATUSES == {0, 1, 6}


def test_the_iterate_split_covers_every_status():
    assert sorted(WITH_ITERATE + list(WITHOUT_ITERATE)) == sorted(IpoptStatus)


@pytest.mark.parametrize("code", WITH_ITERATE)
def test_a_status_with_an_iterate_is_returned(code):
    status = status_or_raise(code)
    assert status is IpoptStatus(code)


@pytest.mark.parametrize(("code", "exception"), WITHOUT_ITERATE.items())
def test_a_status_without_an_iterate_raises(code, exception):
    message = f'Ipopt stopped without a solution. Status {code}: "{IpoptStatus(code).message}"'
    with pytest.raises(exception, match=re.escape(message)):
        status_or_raise(code)


def test_an_unknown_status_raises():
    with pytest.raises(RuntimeError, match="status 7, which this version of YAPSS does not"):
        status_or_raise(7)
