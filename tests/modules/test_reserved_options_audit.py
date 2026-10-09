"""Options that would change the problem YAPSS formulated, or what it reads back, are reserved."""

# standard imports
import re
from pathlib import Path

# third party imports
import pytest

# package imports
from yapss._backend.ipopt_options import RESERVED_IPOPT_OPTIONS, IpoptOptions

DOCS = Path(__file__).parents[2] / "docs" / "user_guide" / "reference" / "ipopt_options.rst"


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("skip_finalize_solution_call", "yes", "a solution Ipopt never wrote"),
        ("grad_f_constant", "yes", "objective gradient is constant"),
        ("jac_c_constant", "yes", "equality constraints is constant"),
        ("jac_d_constant", "yes", "inequality constraints is constant"),
        ("hessian_constant", "yes", "slows convergence"),
        ("mu_target", 1e-4, "perturbed complementarity"),
        ("num_linear_variables", 5, "an order YAPSS decides"),
        ("warm_start_same_structure", "yes", "status -100"),
    ],
)
def test_the_option_is_refused_with_its_reason(name, value, reason):
    pattern = f"'{name}' is managed by YAPSS and cannot be set directly. .*{re.escape(reason)}"
    with pytest.raises(ValueError, match=pattern):
        setattr(IpoptOptions(), name, value)


def test_every_reserved_option_is_documented():
    text = DOCS.read_text(encoding="utf-8")
    missing = [name for name in RESERVED_IPOPT_OPTIONS if f"``{name}``" not in text]
    assert not missing, f"reserved but not documented in {DOCS.name}: {missing}"
