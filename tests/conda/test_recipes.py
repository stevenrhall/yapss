import os
import re
from pathlib import Path

import pytest
import toml
import yaml
from jinja2 import Environment, FileSystemLoader

project_dir = Path(__file__).parents[2]

# Package-name extraction, not exact spec matching. pip and conda spell version
# constraints differently (`numpy<2.5` vs `numpy`, `black[jupyter]>=24.0.0,<25.0.0`
# vs `black[jupyter]==26.5.1`), and some constraints are *deliberately* different
# between the two (see EXPECTED_ONLY_IN_CONDA below) -- comparing full spec
# strings makes this test fail every time a pin is bumped on either side, which
# is exactly the maintenance burden that made it worth disabling in the first
# place. Comparing names only means the test fires for the signal that actually
# matters: a package added or removed on one side and not the other.
_NAME_RE = re.compile(r"^([A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,\-]+\])?)")


def _pkg_name(spec: str) -> str:
    """Extract the package name (with any extras) from a dependency spec string."""
    match = _NAME_RE.match(spec.strip())
    if not match:
        msg = f"Could not parse a package name from dependency spec: {spec!r}"
        raise ValueError(msg)
    return match.group(1).lower()


def _names(specs: set[str]) -> set[str]:
    return {_pkg_name(spec) for spec in specs}


def _spec_by_name(specs: set[str]) -> dict[str, str]:
    """Map package name -> its full spec string, for exact-pin comparisons."""
    return {_pkg_name(spec): spec for spec in specs}


def load_yaml(file: str) -> dict:
    env = Environment(loader=FileSystemLoader(project_dir))
    os.environ["GIT_DESCRIBE_TAG"] = "1.2.3"
    env.globals["environ"] = os.environ
    template = env.get_template(file)
    rendered = template.render({})
    return yaml.safe_load(rendered)


# Deliberate, permanent differences between the pip and conda dependency sets.
# Each entry below is a name-level exception, with the reason it's expected to
# never go away. If a mismatch shows up that *isn't* explained by one of these,
# it's a real sync gap: either a package needs to be added to the conda side
# (or pyproject.toml), or one of these exceptions needs updating.
#
# Present in conda files but not expressed as a pyproject.toml dependency:
EXPECTED_ONLY_IN_CONDA = {
    # pyproject.toml expresses the Python floor via `requires-python`, not as a
    # `dependencies` entry; conda has no equivalent mechanism and must list it.
    "python",
    # Only used on the conda/cyipopt backend; pip installs never import cyipopt
    # (see _private/solver.py's deferred import and its OpenMP-collision comment).
    "cyipopt",
}
# Present in pyproject.toml's core dependencies but not pinned identically in
# conda -- name-only comparison already tolerates different *version* syntax,
# so nothing needs to be listed here for that reason alone. This set is for
# packages conda's recipe is expected to omit outright, if that ever happens.
EXPECTED_ONLY_IN_PYPROJECT: set[str] = set()


def _pyproject_names() -> tuple[set[str], set[str]]:
    """Return (core, optional) package-name sets from pyproject.toml."""
    pyproject = toml.load(project_dir / "pyproject.toml")

    core = _names(set(pyproject["project"]["dependencies"]))

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    doc = _names(set(optional_dependencies["doc"]))
    notebook = _names(set(optional_dependencies["notebook"]))
    # dev's own list includes a self-reference ("yapss[doc,notebook]") to pull in
    # the doc/notebook extras -- not an external package, and doc|notebook already
    # covers what it expands to, so drop anything that names this package itself.
    dev = {
        name for name in _names(set(optional_dependencies["dev"])) if not name.startswith("yapss")
    }
    optional = doc | notebook | dev
    return core, optional


def test_recipes():
    core, optional = _pyproject_names()
    all_ = core | optional | EXPECTED_ONLY_IN_CONDA

    # check that pyproject core and optional dependencies are disjoint
    assert core.isdisjoint(optional)

    # check that environment.yml contains all dependencies, by name
    conda_environment_dependencies = _names(set(load_yaml("conda/environment.yml")["dependencies"]))
    diff = conda_environment_dependencies - all_
    assert diff == set(), f"packages in conda/environment.yml but not in pyproject.toml: {diff}"
    diff = all_ - conda_environment_dependencies - EXPECTED_ONLY_IN_PYPROJECT
    assert diff == set(), f"packages in pyproject.toml but not in conda/environment.yml: {diff}"

    # now check the same two outputs in recipe/meta.yaml
    recipes_meta = load_yaml("conda/recipe/meta.yaml")

    # yapss dependencies in recipe/meta.yaml
    yapss_dependencies = _names(set(recipes_meta["outputs"][0]["requirements"]["run"]))
    expected_yapss = core | EXPECTED_ONLY_IN_CONDA
    diff = yapss_dependencies - expected_yapss
    assert diff == set(), f"packages in recipe/meta.yaml (yapss) but not in pyproject.toml: {diff}"
    diff = expected_yapss - yapss_dependencies - EXPECTED_ONLY_IN_PYPROJECT
    assert diff == set(), f"packages in pyproject.toml but not in recipe/meta.yaml (yapss): {diff}"

    # now do output[1], which is yapss-dev
    yapss_dev_dependencies = _names(set(recipes_meta["outputs"][1]["requirements"]["run"]))
    diff = yapss_dev_dependencies - all_
    assert (
        diff == set()
    ), f"packages in recipe/meta.yaml (yapss-dev) but not in pyproject.toml: {diff}"
    diff = all_ - yapss_dev_dependencies - EXPECTED_ONLY_IN_PYPROJECT
    assert (
        diff == set()
    ), f"packages in pyproject.toml but not in recipe/meta.yaml (yapss-dev): {diff}"

    # now do run at the base level, which should match yapss
    run_dependencies = _names(set(recipes_meta["requirements"]["run"]))
    diff = run_dependencies - yapss_dependencies
    assert (
        diff == set()
    ), f"packages in recipe/meta.yaml (run) but not in recipe/meta.yaml (yapss): {diff}"
    diff = yapss_dependencies - run_dependencies
    assert (
        diff == set()
    ), f"packages in recipe/meta.yaml (yapss) but not in recipe/meta.yaml (run): {diff}"


# `conda/environment-test.yml` deliberately doesn't mirror the full `dev`/`doc`/
# `notebook` extras the way `environment.yml` does -- it's the minimal
# environment the weekly/manual `test-conda` CI job (see
# .github/workflows/ci.yml) needs to install yapss's runtime dependencies plus
# just enough tooling to run pytest, so it's checked against pyproject.toml's
# core dependencies only, not the full `all_` set.
EXPECTED_ONLY_IN_ENV_TEST = {
    # pyproject.toml expresses the Python floor via `requires-python`, not as a
    # `dependencies` entry.
    "python",
    # Only used on the conda/cyipopt backend (see EXPECTED_ONLY_IN_CONDA above).
    "cyipopt",
    # Needed to run the test suite itself.
    "pytest",
    "pytest-cov",
    # Needed to `pip install --no-deps .` from source in this environment.
    "hatchling",
    "hatch-vcs",
    # Needed by this file (test_recipes.py) to parse pyproject.toml and the
    # conda recipe/environment files for these reconciliation checks.
    "toml",
    "pyyaml",
    "jinja2",
}


def test_environment_test_yml():
    core, _optional = _pyproject_names()
    expected = core | EXPECTED_ONLY_IN_ENV_TEST

    env_test_dependencies = _names(set(load_yaml("conda/environment-test.yml")["dependencies"]))
    diff = env_test_dependencies - expected
    assert diff == set(), f"packages in conda/environment-test.yml but not accounted for: {diff}"
    diff = core - env_test_dependencies
    assert (
        diff == set()
    ), f"packages in pyproject.toml core but not in conda/environment-test.yml: {diff}"


# --- dev-tool pinning policy -----------------------------------------------
#
# `pyproject.toml`'s `dev` extra pins tools like black/isort/ruff/mypy/tox
# exactly (`==`), so contributors and CI get identical lint/format output.
# That reasoning carries over unchanged to `conda/environment.yml`: it's a
# local `conda env create -f` target for a contributor's own dev environment,
# never published or composed with anyone else's environment, so an exact pin
# costs nothing and buys the same reproducibility.
#
# It does *not* carry over to `conda/recipe/meta.yaml`'s `yapss-dev` output.
# That recipe is published to conda-forge and installed via
# `conda install conda-forge::yapss-dev` into arbitrary users' environments,
# alongside whatever else they have. An exact `==` pin on a dev tool there is
# exactly the pattern conda-forge's own linting pushes back on: it forces
# conda's solver to find that one exact build in every environment yapss-dev
# joins, a common source of unsatisfiable-environment conflicts, and it forces
# a yapss-dev rebuild every time a pin moves in pyproject.toml even if nothing
# about yapss itself changed. So the same tools should appear in
# `meta.yaml`'s `yapss-dev` run requirements *unpinned*, or with at most a
# loose compatible-release range guarding a known incompatibility -- not
# mirroring pyproject's exact version.
#
# This is a different kind of pin than `numpy<2.5` or the `casadi` ceiling
# elsewhere in these files: those are defensive ceilings blocking a specific
# known-bad range, not "match what we're using right now," and conda-forge
# tolerates that distinction.
def test_dev_tool_pinning_policy():
    pyproject = toml.load(project_dir / "pyproject.toml")
    dev_specs = _spec_by_name(set(pyproject["project"]["optional-dependencies"]["dev"]))

    # The dev tools this project exact-pins in pyproject.toml's dev extra.
    exact_pinned = {name: spec for name, spec in dev_specs.items() if "==" in spec}
    assert exact_pinned, "expected at least one exact-pinned dev tool in pyproject.toml's dev extra"

    env_specs = _spec_by_name(set(load_yaml("conda/environment.yml")["dependencies"]))
    recipes_meta = load_yaml("conda/recipe/meta.yaml")
    yapss_dev_specs = _spec_by_name(set(recipes_meta["outputs"][1]["requirements"]["run"]))

    for name, pyproject_spec in exact_pinned.items():
        # environment.yml: exact pin, identical to pyproject.toml's.
        assert (
            name in env_specs
        ), f"{name!r} (exact-pinned in pyproject.toml) is missing from environment.yml"
        assert env_specs[name] == pyproject_spec, (
            f"{name!r} pin drifted: pyproject.toml has {pyproject_spec!r}, "
            f"conda/environment.yml has {env_specs[name]!r} -- environment.yml should mirror "
            f"pyproject.toml's dev extra exactly (see policy comment above)"
        )

        # recipe/meta.yaml (yapss-dev): present, but never exact-pinned.
        assert (
            name in yapss_dev_specs
        ), f"{name!r} is missing from recipe/meta.yaml's yapss-dev run requirements"
        assert "==" not in yapss_dev_specs[name], (
            f"{name!r} is exact-pinned in recipe/meta.yaml's yapss-dev output "
            f"({yapss_dev_specs[name]!r}) -- published conda recipes shouldn't exact-pin dev "
            f"tools (see policy comment above); use no constraint or a loose range instead"
        )


if __name__ == "__main__":
    pytest.main([__file__])
