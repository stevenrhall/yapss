# Release Checklist

A reusable checklist for cutting a YAPSS release. This describes the steady-state
process, not any one release's history.

**How to keep this current, without churn:** update this file when a step
changes for good (a tool gets replaced, a step turns out to be unnecessary, a
new step is discovered to be required every time). Don't edit it just because
one release did something slightly different, worked around a one-off
problem, or skipped a step for a good reason specific to that release — note
those in that release's own CHANGELOG entry, commit messages, or PR
description instead. If the same "one-off" comes up twice, that's the signal
it belongs here.

---

## 1. Dependencies

- [ ] Check for any new reactive ceilings needed (a dependency broke since
      last release) or ceilings that can now be lifted (the underlying issue
      was fixed upstream).
- [ ] Confirm `pyproject.toml` is the source of truth for pip dependencies —
      this is what the conda-forge feedstock will eventually mirror by hand.
- [ ] Manually sync `conda/environment-test.yml` and `conda/recipe/meta.yaml`
      with `pyproject.toml` if dependencies changed.

## 2. Changelog

- [ ] Update `CHANGELOG.md` with an entry for the new version, dated.
- [ ] If the pip and conda builds differ in any user-visible way (solver
      backend, supported Python range, etc.), say so explicitly — it's the
      difference that generates confusing bug reports.

## 3. Documentation review (local build)

Do this *before* tagging — it's much cheaper to fix docs pre-release than to
carry a stale RTD-published version forward (RTD builds each version from its
git tag, and the published content is not edited after the fact by moving
the tag).

- [ ] Review `docs/user_guide/index.md` (the RTD landing page — `README.md`
      is generated from it via `make readme`, don't edit `README.md`
      directly). Check supported Python versions, install commands, and any
      hardcoded version/tag references are current.
- [ ] Check `LICENSE`'s copyright year range is current (it's a static legal
      file, not auto-generated — bump it manually, e.g. "2021-2026"). The
      Sphinx config copyright notices (`docs/*/conf.py`) compute the end
      year dynamically, so those don't need manual attention.
- [ ] Run `make clean` and `make docs` (or `make view-docs` to build and open it in a browser) and
      review the rendered output — reference pages, examples, tutorial — for
      anything this release's changes made stale or incorrect. Source review
      alone misses rendering problems.
- [ ] For every user-visible change in this release's `CHANGELOG.md` entry,
      confirm the docs actually reflect it — a change that's changelogged but
      not documented is easy to miss until a user hits it.
- [ ] `make docs` regenerates `README.md` as a side effect (`docs` depends on
      `readme` in the Makefile) -- after any `docs/user_guide/index.md` edit,
      confirm `README.md` came out changed as expected and commit it alongside.
- [ ] Run `make linkcheck` and triage every non-`ok` result. `redirect` is
      normally fine -- DOI resolvers redirecting to the publisher's canonical
      URL, GitHub issue-template links redirecting to a login page for an
      unauthenticated request, and the readthedocs.io -> /en/stable/ redirect
      are all expected -- but check that the destination is actually the
      right page, not a squatted domain or something unrelated. `-ignored-`
      means Sphinx's own linkcheck config already excluded it; no action
      unless that config looks wrong. `broken` needs a look, but a `403
      Forbidden` from a major academic publisher (ACM, Wiley, Oxford
      Academic, SIAM, AIAA, etc.) usually means the site is blocking
      automated requests, not that the citation is actually dead --
      spot-check a couple by hand in a real browser before spending time
      trying to fix them.
- [ ] Push to the release branch, then view `README.md` on that branch's
      GitHub code page and confirm it renders correctly. In particular,
      check that relative links resolve to their repo-page counterparts,
      not their docs-page counterparts -- e.g. a "Contributing" link should
      point to `CONTRIBUTING.md` at the repo root, not to the
      `contributing.rst` page under `docs/`. GitHub renders `README.md`'s
      links relative to the repo root, not `docs/user_guide/`, so a link
      that's correct in the Sphinx-rendered docs can still be wrong here.
      If a link is wrong, the `make readme` Makefile logic needs fixing,
      not just the link.

## 4. CI

- [ ] Review `.github/workflows/*.yml` for correctness and needed updates --
      stale comments referencing things fixed elsewhere this release, Python
      versions that no longer match the supported range, jobs that should be
      added or retired, and so on. CI config drifts quietly the same way the
      conda files do, and nothing else in this checklist catches it.
- [ ] Green CI on the release branch: full test matrix, lint, docs, and (if
      dependencies touching the conda stack changed) a manual
      `workflow_dispatch` run of the conda test job.
- [ ] Before merging a branch that changes workflow trigger config (`on:`
      blocks), validate with a manual `workflow_dispatch` run *from that
      branch* as a belt-and-suspenders check. Confirmed on the v0.1.1 PR:
      GitHub reliably picks up trigger changes on the PR's own checks as long
      as the workflow *file* already exists on the default branch — the risk
      case is a workflow file that's entirely new to the repo, which won't
      fire on `pull_request` until it's merged.

## 5. RTD build (hosted)

- [ ] Once CI is green, trigger a trial build on Read the Docs itself for the
      release branch/PR (not just local `make docs`/tox) and confirm it's
      clean. RTD's real environment differs from local in ways that matter
      (real internet access for intersphinx, its own pinned toolchain) and
      requires RTD project admin access, which not every contributor has.
      This is a pre-merge sanity check, distinct from the post-publish
      `stable`-resolves-correctly check in §10.

## 6. Build verification

Build from a fresh `git clone` of the release branch into a temp dir, not the
local working tree — a local build can hide files that aren't actually
tracked in git (missing from packaging config) or stale artifacts left over
from previous builds/editable installs.

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
- [ ] Spot-check the sdist and wheel file listings (`tar tzf dist/*.tar.gz` /
      `unzip -l dist/*.whl`) for missing package data or accidentally-included
      dev/test cruft.

## 7. Merge to main

YAPSS uses a PR-into-`main`-then-tag model — no dedicated release branch.
`hatch-vcs` derives the published version from the tag, and a single `main` +
tags is simpler than maintaining release branches, which would only earn
their keep if YAPSS needed to maintain multiple release lines in parallel
(e.g. hotfixing an old minor after a newer one shipped) — not a current need.

- [ ] Open a PR from the release branch into `main` (don't push directly) —
      this exercises CI and the RTD PR-preview build (§4, §5) against the
      actual merge target.
- [ ] Squash merge, with a commit message drawn from the PR description or
      `CHANGELOG.md` entry rather than GitHub's default (which concatenates
      every commit subject from the branch).
- [ ] Delete the head branch after merge — safe to do; the squashed commit is
      already permanent in `main`'s history, and the PR page retains the full
      pre-squash commit history regardless.
- [ ] Confirm CI is green on `main` itself (a fresh `push` event on the real
      trigger config, not just the branch's).

## 8. Tag & publish

Publishing is automated by `.github/workflows/publish.yml`, triggered by the
tag push below. It builds once, publishes that same dist to TestPyPI,
installs it from TestPyPI into a clean environment and runs a real example,
and then -- only after that succeeds and a human approves it in the Actions
UI -- publishes the identical dist to PyPI. No local `twine`/`build`
commands are needed; nothing to fat-finger and no way to burn a version
number by hand.

- [ ] Dry run: tag and push a throwaway pre-release tag (e.g. `vX.Y.Zrc1`) to
      exercise the publish workflow end-to-end -- build, TestPyPI publish,
      install smoke test -- without approving the final `pypi` deployment.
      Delete the tag locally and on the remote afterward so it doesn't
      linger:
      ```
      git tag -d vX.Y.Zrc1
      git push origin :refs/tags/vX.Y.Zrc1
      ```
      Stopping at TestPyPI is enough for routine confidence-building: it
      doesn't touch conda either way, since the conda-forge autotick bot
      only tracks stable PyPI releases and polls PyPI independently of
      anything in this repo, pre-release or not. Skip the dry run entirely
      if the workflow has already been exercised successfully on a recent
      release and nothing about the publish pipeline (`publish.yml`, the
      trusted-publisher registrations, or the `pypi` environment's
      protection rule) has changed since.
- [ ] The first time this workflow is used, and again after any change to
      it, go one step further: actually approve the `pypi` deployment for
      the rc tag, so the rc is published for real, then run a plain
      `pip install yapss==X.Y.Zrc1` with no `--index-url` flags. TestPyPI
      only proves the package installs with `--extra-index-url` pointed at
      it, not that a real, unflagged `pip install` resolves correctly --
      and that gap is exactly what a first real run of a new or changed
      workflow is most likely to expose. This step is not free -- a
      published pre-release, like a stable one, can't be deleted or
      re-uploaded if something's wrong with it, only superseded by `rc2` --
      so it is deliberately not the routine case: the risk is publishing an
      rc that turns out to be broken, not the real `X.Y.Z`.
- [ ] Tag `vX.Y.Z` and push. `hatch-vcs` derives the version from the tag,
      and the tag push triggers the publish workflow.
- [ ] Watch the workflow run. Once the TestPyPI publish and install-smoke-test
      jobs are green, review them before approving -- the approval step is
      the last chance to catch a problem before the PyPI publish becomes
      permanent.
- [ ] Approve the `pypi` deployment in the Actions UI to let the final job
      run.
- [ ] Confirm the new version appears on pypi.org.

## 9. Conda

- [ ] Wait for the conda-forge autotick bot PR (hours, not immediate). It
      updates version and sha256 and resets the build number to 0, but does
      **not** sync dependency changes — apply those by hand if dependencies
      changed this release.
- [ ] Merge the bot PR from a fork. conda-forge rejects PRs from branches on
      the feedstock repo itself, even from maintainers.
- [ ] Verify: `conda create -n check -c conda-forge yapss`, then run the
      isoperimetric example.

## 10. Post-publish docs checks

- [ ] Confirm `stable` on Read the Docs resolves to the new version (RTD
      picks the highest semver tag by default, but check it wasn't manually
      pinned at some point).
- [ ] Spot-check the published RTD page itself, not just the source — a
      clean build doesn't guarantee the rendered output looks right.
- [ ] If the previous release had a known-broken install and a docs banner
      or notice was added for it, remove or update that notice now.
