# Release Checklist

Use this checklist for each YAPSS release. It defines the standard release
process rather than the history of a specific release.

Update this file when the standard process changes, such as when a tool is
replaced or a recurring step is added or removed. Record release-specific
deviations in the applicable `CHANGELOG.md` entry, commit messages, or pull
request description. Add a deviation to this checklist if it becomes part of
the recurring process.

---

## 1. Dependencies

- [ ] Review dependency ceilings. Add a ceiling when an upstream regression
      requires one, and remove it when the underlying issue has been resolved.
- [ ] Confirm that `pyproject.toml` remains the source of truth for pip
      dependencies. The conda-forge feedstock is synchronized from it
      manually.
- [ ] If dependencies changed, manually synchronize the applicable conda
      mirrors with `pyproject.toml`: `conda/environment-test.yml` for the test
      environment, `conda/environment.yml` for the contributor environment,
      and `conda/recipe/meta.yaml` for the package and development outputs.

## 2. Changelog

- [ ] Update `CHANGELOG.md` with an entry for the new version, dated.
- [ ] Update `version` and `date-released` in `CITATION.cff`, and the version in
      the citation shown under "Citing YAPSS" in `docs/user_guide/index.md`.
      Zenodo reads `CITATION.cff` at the release commit for the archive's
      metadata, so it has to be right before tagging.
- [ ] Document any user-visible differences between the pip and conda builds,
      such as the solver backend or supported Python range.

## 3. Documentation review (local build)

Complete the local documentation review before tagging. Read the Docs builds
each version from its Git tag; moving a tag does not revise already-published
content.

- [ ] Review `docs/user_guide/index.md`, the Read the Docs landing page.
      `README.md` is generated from this file by `make readme` and should not
      be edited directly. Confirm that supported Python versions, installation
      commands, and hardcoded version or tag references are current.
- [ ] Confirm that the copyright year range in `LICENSE` is current. This is a
      static file and must be updated manually, for example to "2021-2026".
      The Sphinx configurations in `docs/*/conf.py` calculate the end year
      dynamically and require no corresponding update.
- [ ] Run `make clean` and `make docs`, or use `make view-docs` to build and
      open the documentation. Review the rendered reference pages, examples,
      and tutorial for stale content and rendering problems.
- [ ] Confirm that the documentation reflects every user-visible change in
      this release's `CHANGELOG.md` entry.
- [ ] `make docs` regenerates `README.md` because the Makefile's `docs` target
      depends on `readme`. After editing `docs/user_guide/index.md`, confirm
      that the generated `README.md` contains the expected changes and include
      it in the same commit.
- [ ] Run `make linkcheck` and review every non-`ok` result. Expected redirects
      include DOI resolvers to publisher pages, GitHub issue-template links to
      a login page for unauthenticated requests, and readthedocs.io to
      `/en/stable/`; confirm that each destination is correct. An `-ignored-`
      result requires no action unless the Sphinx linkcheck exclusion is
      incorrect. Investigate each `broken` result. A `403 Forbidden` response
      from an academic publisher commonly indicates that automated requests
      are blocked rather than that the citation is unavailable; verify a
      representative sample in a browser.
- [ ] One `broken` result is expected at this point in every release: the
      `CHANGELOG.md` link definition for the version being released points at a
      GitHub compare against its tag, and that tag is not pushed until section 8.
      It is reported broken by `make linkcheck` only; Read the Docs builds HTML
      and does not check links, so the sole effect there is a dead link on the
      rendered changelog until the tag exists. Confirm the 404 is that link and
      no other.
- [ ] Push the release branch, then view `README.md` on the branch's GitHub
      code page and confirm that it renders correctly. Relative links must
      resolve to repository-page targets rather than documentation-page
      targets. For example, a "Contributing" link should point to the
      repository-root `CONTRIBUTING.md`, not `docs/user_guide/contributing.rst`.
      GitHub resolves README links from the repository root, while Sphinx
      resolves them from `docs/user_guide/`. Correct link-generation errors in
      the Makefile's `readme` target.

## 4. CI

- [ ] Review `.github/workflows/*.yml` for correctness and required updates.
      Check comments, supported Python versions, action versions, job coverage,
      permissions, and jobs that should be added or retired.
- [ ] Confirm green CI on the release branch: full test matrix, lint, docs,
      and the `test-conda` job that runs automatically on pull requests. If
      contributor dependencies or `conda/environment.yml` changed, manually
      dispatch `test-conda-dev-env`, which otherwise runs only on its weekly
      schedule.
- [ ] Before merging a branch that changes workflow trigger configuration
      (`on:` blocks), run the workflow manually with `workflow_dispatch` from
      that branch. GitHub applies trigger changes to pull-request checks when
      the workflow file already exists on the default branch, as confirmed on
      the v0.1.1 pull request. A new workflow file does not run on
      `pull_request` until it is present on the default branch.

## 5. RTD build (hosted)

- [ ] Once CI is green, trigger a Read the Docs build for the release branch or
      pull request and confirm that it succeeds. The hosted environment has
      different network access and toolchain constraints from local
      `make docs` or tox builds and requires Read the Docs project
      administration access. This pre-merge check is separate from the
      post-publish `stable` check in §9.

## 6. Build verification

Build from a fresh clone of the release branch in a temporary directory. A
build from the working tree can include untracked files or stale artifacts
from previous builds and editable installations.

- [ ] Clean build in a clean environment:
      ```
      git clone -b <release-branch> --single-branch https://github.com/stevenrhall/yapss /tmp/yapss-build-check
      cd /tmp/yapss-build-check
      python -m venv .venv && source .venv/bin/activate
      pip install build
      python -m build
      pip install dist/*.whl
      python -m yapss.examples.isoperimetric
      ```
- [ ] Review the sdist and wheel file listings (`tar tzf dist/*.tar.gz` and
      `unzip -l dist/*.whl`) for missing package data and unintended
      development or test files. Compare the wheel's `METADATA` with the
      sdist's `PKG-INFO`, and confirm the version, Python requirement,
      dependencies, license files, and package data in both artifacts.

## 7. Merge to main

YAPSS merges release changes into `main` before tagging and does not maintain
dedicated release branches. `hatch-vcs` derives the published version from the
tag. Separate release branches would be appropriate only if multiple release
lines needed concurrent maintenance.

- [ ] Open a pull request from the release branch into `main`; do not push
      directly. This runs CI and the Read the Docs preview (§4 and §5) against
      the merge target.
- [ ] Squash merge, with a commit message drawn from the PR description or
      `CHANGELOG.md` entry rather than GitHub's default (which concatenates
      every commit subject from the branch).
- [ ] Confirm that GitHub deleted the head branch automatically after the
      merge; otherwise delete it manually. The squashed commit remains in
      `main`, and the pull-request page retains the pre-squash commit history.
- [ ] Confirm that CI succeeds on `main` after the merge's `push` event.
- [ ] Update local `main`, delete the merged local branch, and run
      `git remote prune origin` to remove stale remote-tracking references.
- [ ] Confirm that Read the Docs does not retain an active version for the
      deleted branch. Configure an RTD automation rule with the `Delete
      version` action for deleted branches so this cleanup normally occurs
      automatically. Pull-request previews are separate and expire according
      to RTD's retention policy.

## 8. Tag & publish

Publishing is automated by `.github/workflows/publish.yml`, triggered by the
tag push below. The workflow builds one distribution, publishes it to
TestPyPI, installs it in a clean environment, and runs an example. After the
smoke test succeeds and a reviewer approves the protected `pypi` environment,
the workflow publishes the same distribution to PyPI. Local publishing
commands are not used.

- [ ] If a release-candidate dry run is required, create and push a temporary
      pre-release tag such as `vX.Y.Zrc1`. Confirm that the build, TestPyPI
      publication, and installation smoke test succeed. TestPyPI distributions
      cannot be replaced with the same version and filenames; if the artifacts
      require correction, use a new tag such as `vX.Y.Zrc2` rather than
      rerunning the complete workflow for the existing version. The dry run
      may be omitted if a recent release exercised the same workflow and
      neither `publish.yml`, the trusted-publisher registrations, nor the
      `pypi` environment protection rule has changed.
- [ ] On the first use of the workflow and after any subsequent workflow
      change, leave the release-candidate workflow waiting at the protected
      `pypi` environment, review the preceding jobs, and approve the
      production deployment. Then run `pip install yapss==X.Y.ZrcN` without
      index options. TestPyPI verifies installation only when configured as an
      additional index; publishing the release candidate verifies normal PyPI
      resolution. For a routine dry run that does not require this production
      test, cancel the workflow at the approval gate instead.
- [ ] After completing the applicable release-candidate path, delete its local
      and remote Git tags:
      ```
      git tag -d vX.Y.ZrcN
      git push origin :refs/tags/vX.Y.ZrcN
      ```
      Deleting the Git tag does not remove a distribution already published to
      PyPI or TestPyPI. Leave a production release candidate published, or yank
      it with a reason after the final release; do not delete it as routine
      cleanup. The conda-forge autotick bot tracks stable PyPI releases and
      does not act on pre-releases.
- [ ] Fetch `main`, confirm that local and remote `main` point to the approved
      release commit, and confirm that `vX.Y.Z` is unused. Create and push the
      final tag. `hatch-vcs` derives the version from the tag, and the tag push
      triggers the publish workflow.
- [ ] Before approving production publication, confirm that CI for the exact
      tag succeeds, the tagged Read the Docs build succeeds, and the Publish
      workflow's build, TestPyPI publication, installation, and smoke-test jobs
      succeed. Review the TestPyPI version, dependency constraints, wheel, and
      sdist.
- [ ] Approve the `pypi` deployment in the Actions UI to run the final job.
- [ ] Confirm that the new version appears on pypi.org and that the production
      wheel and sdist SHA-256 values match the artifacts tested on TestPyPI.
- [ ] Create the GitHub Release for the tag, with notes drawn from the
      changelog entry:
      ```
      gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <notes>
      ```
      A tag push alone does not archive anything: Zenodo's GitHub integration
      acts on *published releases*. Zenodo then mints the version DOI under the
      project's concept DOI (the one in `CITATION.cff`) within a few minutes;
      confirm the new version appears on the Zenodo record.
- [ ] If this release supersedes a release with serious installation,
      correctness, crash, or security defects, review whether the affected
      release should be yanked. Prefer yanking with a clear reason to deleting
      a release, which breaks exact pins and is irreversible.

## 9. Post-publish docs checks

- [ ] Confirm that `stable` on Read the Docs resolves to the new version. Read
      the Docs normally selects the highest semantic-version tag; verify that
      `stable` has not been configured manually.
- [ ] Review representative pages in the published documentation. A successful
      build does not confirm that the rendered output is correct.
- [ ] If the previous release had a known-broken install and a docs banner
      or notice was added for it, remove or update that notice now.

## 10. Conda

Conda-forge automation may lag the PyPI release by several days; it does not
block the post-publish documentation checks in section 9.

- [ ] Wait for the conda-forge autotick bot pull request. The bot updates the
      version and SHA-256 value and resets the build number to 0, but does
      **not** synchronize dependency changes; apply those manually when
      required.
- [ ] Merge the bot PR from a fork. conda-forge rejects PRs from branches on
      the feedstock repo itself, even from maintainers.
- [ ] Verify: `conda create -n check -c conda-forge yapss`, then run the
      isoperimetric example.
