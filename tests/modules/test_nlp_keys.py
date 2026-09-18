"""The NLP key maps: each NLP entry is keyed by the view it lives in.

`nlp_variable_keys` and `nlp_constraint_keys` key every entry of the NLP decision variable
and constraint vectors as ``(phase, view, component, position)``. These tests check the
keys against the integer structures directly -- looking a key's view up by name, not
through the code that wrote it -- so a key can only pass by naming the entry it sits on.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from yapss import Problem
from yapss._private.layout import SPECTRAL_METHODS, problem_layout
from yapss._private.structure import (
    get_nlp_cf_structure,
    get_nlp_dv_structure,
    nlp_constraint_keys,
    nlp_variable_keys,
)
from yapss._private.types_ import CFViewName, DVViewName

SCALAR_VIEWS = {"q", "s", "integral", "discrete"}
"""Views whose components are single entries of one array, indexed by component."""

SINGLE_VIEWS = {"t0", "tf", "duration"}
"""Views that are one length-1 array."""


def problem(method):
    """Two phases, two segments in the first, with parameters, integrals, paths, discretes."""
    ocp = Problem(name="keys", nx=[2, 1], nu=[1, 2], nq=[1, 0], nh=[1, 1], ns=2, nd=3)
    ocp.spectral_method = method
    ocp.mesh.phase[0].collocation_points = (3, 4)
    ocp.mesh.phase[0].fraction = (0.5, 0.5)
    return ocp


def lookup(structure, extra, p, view, i):
    """Return the component array a key names, reading the view by attribute name."""
    owner = structure if view in extra else structure.phase[p]
    array = getattr(owner, view)
    if view in SCALAR_VIEWS:
        return array[i : i + 1]
    if view in SINGLE_VIEWS:
        assert i == 0
        return array
    return array[i]


@pytest.mark.parametrize("method", SPECTRAL_METHODS)
def test_every_variable_key_names_its_own_entry(method):
    ocp = problem(method)
    keys = nlp_variable_keys(ocp._to_spec())
    dv = get_nlp_dv_structure(ocp._to_spec(), np.int64)
    dv.z[:] = np.arange(dv.z.size)
    assert keys.shape == dv.z.shape
    for k, (p, view, i, j) in enumerate(keys):
        assert view in get_args(DVViewName)
        assert lookup(dv, {"s"}, p, view, i)[j] == k
    unused = {"xs"} if method != "lgl" else set()
    assert {key[1] for key in keys} == set(get_args(DVViewName)) - unused


@pytest.mark.parametrize("method", SPECTRAL_METHODS)
def test_every_constraint_key_names_its_own_entry(method):
    ocp = problem(method)
    keys = nlp_constraint_keys(ocp._to_spec())
    cf = get_nlp_cf_structure(ocp._to_spec(), np.int64)
    cf.c[:] = np.arange(cf.c.size)
    assert keys.shape == cf.c.shape
    for k, (p, view, i, j) in enumerate(keys):
        assert view in get_args(CFViewName)
        assert lookup(cf, {"discrete"}, p, view, i)[j] == k
    unused = {"lg_defect"} if method != "lg" else set()
    assert {key[1] for key in keys} == set(get_args(CFViewName)) - unused


@pytest.mark.parametrize("method", SPECTRAL_METHODS)
def test_aliased_state_views_read_the_state_key_at_its_storage_position(method):
    """x0 and xf are aliases into x, and a position is a storage position, not a time index."""
    ocp = problem(method)
    dv = get_nlp_dv_structure(ocp._to_spec(), object)
    dv.z[:] = nlp_variable_keys(ocp._to_spec())
    for p, layout in enumerate(problem_layout(ocp._to_spec())):
        for i in range(ocp.nx[p]):
            assert dv.phase[p].x0[i] == (p, "x", i, layout.x0_position)
            assert dv.phase[p].xf[i] == (p, "x", i, layout.xf_position)


def test_a_problem_without_phases_keys_parameters_and_discretes_with_phase_0():
    ocp = Problem(name="no phases", nx=[], nu=[], ns=2, nd=1)
    assert nlp_variable_keys(ocp._to_spec()).tolist() == [(0, "s", 0, 0), (0, "s", 1, 0)]
    assert nlp_constraint_keys(ocp._to_spec()).tolist() == [(0, "discrete", 0, 0)]
