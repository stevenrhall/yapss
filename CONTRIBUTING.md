# Contributing to YAPSS

Thank you for considering contributing to YAPSS. Your help is vital for improving and maintaining
this project.

## How to Contribute

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Code Contributions](#code-contributions)
- [Coding Guidelines](#coding-guidelines)
- [Licensing](#licensing)
- [Communication](#communication)

### Reporting Bugs

If you find a bug, please report it by
[opening an issue](https://github.com/stevenrhall/yapss/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=%5BBUG%5D+%3CDescriptive+Title%3E).

- **Use a clear and descriptive title** for the issue.
- **Describe the bug**: Provide a clear and concise description of what the bug is.
- **Steps to reproduce**: Explain the steps to reproduce the issue.
- **Expected behavior**: Describe what you expected to happen.
- **Screenshots**: If applicable, add screenshots to help explain the problem.
- **Environment**: Include details about your setup (e.g., OS, Python version, dependencies).

### Suggesting Enhancements

If you have an idea for an enhancement, please
[open an issue](https://github.com/stevenrhall/yapss/issues/new?&labels=enhancement&template=enhancement_template.md&title=%5BENHANCEMENT%5D+%3CDescriptive+Title%3E)
and describe:

- **Use a clear and descriptive title** for the issue.
- **Describe the enhancement**: Provide a clear and concise description of what you want to happen.
- **Explain why**: Explain why this enhancement would be useful.

### Code Contributions

1. **Fork and clone the repository**, then create a branch from `main`.
2. **Install dependencies**:
   - Create and activate a virtual environment (e.g., `venv`).  
   - Install the required dependencies by running:  
     ```bash
     pip install ".[dev]"
     ```  
3. **Write tests**: 
   - Add tests for your new code in the `tests/` directory (using `pytest`).  
   - Ensure your tests cover edge cases and validate the functionality of your contribution.  
4. **Run checks**:
   - Run the checks relevant to your changes before opening a pull request.
   - CI runs the complete `tox` matrix across all supported Python versions, and it must
     pass before a pull request can be merged. Contributors with all supported Python
     versions installed can reproduce the full suite locally with:
     ```bash
     tox
     ```
   - Ensure no type errors or linting issues are present before submitting your PR.
5. **Submit a pull request (PR)**:  
   - Create a PR with a clear title and description of your changes.  
   - A project maintainer will review your PR and may request changes before merging.  
   - Ensure your branch is up-to-date with `main` before submitting the PR. 

### Documentation Sources

The project has two documentation copy directions:

- `README.md` is generated from `docs/user_guide/index.md`; edit the user-guide source,
  then run `make readme`.
- `CHANGELOG.md` and `CONTRIBUTING.md` are maintained at the repository root and copied
  into the user guide during its build; edit the root files, not their copies under
  `docs/user_guide/`.

### Coding Guidelines

- Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.
- Use the versions of [Black](https://github.com/psf/black),
  [isort](https://pycqa.github.io/isort/), and [Ruff](https://docs.astral.sh/ruff/) pinned by
  the project's development setup (`pip install ".[dev]"`); do not rely on globally installed
  versions.
- Write clear and descriptive commit messages.

### Licensing

By contributing to YAPSS, you agree that your contributions will be licensed under the same license
as the project (MIT License).

### Communication

If you need help or have any questions, feel free to reach out by [opening an issue](https://github.com/stevenrhall/yapss/issues).

Thank you for contributing to YAPSS!
