import os
from pathlib import Path

import pytest
import toml
import yaml
from jinja2 import Environment, FileSystemLoader

project_dir = Path(__file__).parents[2]


def load_yaml(file):
    env = Environment(loader=FileSystemLoader(project_dir))
    os.environ["GIT_DESCRIBE_TAG"] = "1.2.3"
    env.globals["environ"] = os.environ
    template = env.get_template(file)
    rendered = template.render({})
    return yaml.safe_load(rendered)


def test_recipes():
    # get dependencies and optional dependencies from pyproject.toml
    pyproject = toml.load(project_dir / "pyproject.toml")

    # core dependencies
    core = set(pyproject["project"]["dependencies"]) - {"mseipopt"}

    # optional dependencies
    optional_dependencies = pyproject["project"]["optional-dependencies"]
    doc = set(optional_dependencies["doc"])
    notebook = set(optional_dependencies["notebook"])
    dev = set(optional_dependencies["dev"])
    optional = doc | notebook | dev
    all_ = core | optional | {"cyipopt"}

    # check if pyproject core and optional dependencies are disjoint
    assert core.isdisjoint(optional)

    # check that environment.yml contains all dependencies
    conda_environment_dependencies = set(load_yaml("conda/environment.yml")["dependencies"])
    diff = conda_environment_dependencies - all_
    print(f"packages in conda/environment.yml but not in pyproject.toml: {diff}")
    # should only contain python
    assert len(diff) == 1
    assert list(diff)[0].startswith("python")

    # check that pyproject contains all dependencies in environment.yml
    diff = all_ - conda_environment_dependencies
    print(f"packages in pyproject.toml but not in conda/environment.yml: {diff}")
    assert diff == set()

    # now check the same two outputs in recipes/meta.yaml
    recipes_meta = load_yaml("conda/recipe/meta.yaml")

    # yapss dependencies in recipes/meta.yaml
    yapss_dependencies = set(recipes_meta["outputs"][0]["requirements"]["run"])
    diff = yapss_dependencies - core - {"cyipopt"}
    print(f"packages in recipe/meta.yaml (yapss) but not in pyproject.toml: {diff}")
    # should only contain python
    assert len(diff) == 1
    assert list(diff)[0].startswith("python")

    # check that pyproject core contains all dependencies in recipe/meta.yaml
    diff = core - yapss_dependencies
    print(f"packages in pyproject.toml but not in recipe/meta.yaml (yapss): {diff}")
    assert diff == set()

    # now do output[1] which is yapss-dev
    yapss_dev_dependencies = set(recipes_meta["outputs"][1]["requirements"]["run"])
    diff = yapss_dev_dependencies - all_
    print(f"packages in recipe/meta.yaml (yapss-dev) but not in pyproject.toml: {diff}")
    # should only contain python
    assert len(diff) == 1
    assert list(diff)[0].startswith("python")

    # check that pyproject contains all dependencies in recipe/meta.yaml
    diff = all_ - yapss_dev_dependencies
    print(f"packages in pyproject.toml but not in recipe/meta.yaml (yapss-dev): {diff}")
    assert diff == set()

    # now do run at base level, which should be the same as yapss
    run_dependencies = set(recipes_meta["requirements"]["run"])
    diff = run_dependencies - yapss_dependencies
    print(f"packages in recipe/meta.yaml (run) but not in recipe/meta.yaml (yapss): {diff}")
    assert diff == set()
    diff = yapss_dependencies - run_dependencies
    print(f"packages in recipe/meta.yaml (yapss) but not in recipe/meta.yaml (run): {diff}")
    assert diff == set()


if __name__ == "__main__":
    pytest.main([__file__])
