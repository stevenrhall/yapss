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
      with `pyproject.toml` if dependencies changed. No automated drift check
      exists yet (see `TOOLING_PLAN.md` if that's still true).

## 2. Changelog

- [ ] Update `CHANGELOG.md` with an entry for the new version, dated.
- [ ] If the pip and conda builds differ in any user-visible way (solver
      backend, supported Python range, etc.), say so explicitly — it's the
      difference that generates confusing bug reports.

## 3. Documentation review

Do this *before* tagging — it's much cheaper to fix docs pre-release than to
carry a stale RTD-published version forward (RTD builds each version from its
git tag; the published content can't be edited after the fact without moving
the tag, which isn't done — see `TOOLING_PLAN.md` if that's still the
policy).

- [ ] Review `docs/user_guide/index.md` (the RTD landing page — `README.md`
      is generated from it via `make readme`, don't edit `README.md`
      directly). Check supported Python versions, install commands, and any
      hardcoded version/tag references are current.
- [ ] Check `LICENSE`'s copyright year range is current (it's a static legal
      file, not auto-generated — bump it manually, e.g. "2021-2026"). The
      Sphinx config copyright notices (`docs/*/conf.py`) compute the end
      year dynamically, so those don't need manual attention.
- [ ] `make docs` (or `make view-docs` to build and open it in a browser) and
      review the rendered output — reference pages, examples, tutorial — for
      anything this release's changes made stale or incorrect. Source review
      alone misses rendering problems.
- [ ] For every user-visible change in this release's `CHANGELOG.md` entry,
      confirm the docs actually reflect it — a change that's changelogged but
      not documented is easy to miss until a user hits it.
- [ ] Run `make readme` after any `docs/user_guide/index.md` edits, and
      commit the regenerated `README.md` alongside.

## 4. CI

- [ ] Green CI on the release branch: full test matrix, lint, docs, and (if
      dependencies touching the conda stack changed) a manual
      `workflow_dispatch` run of the conda test job.
- [ ] Before merging a branch that changes workflow trigger config (`on:`
      blocks), validate with a manual `workflow_dispatch` run *from that
      branch* rather than trusting the PR's own check run. GitHub only runs a
      `pull_request`-triggered workflow reliably once that trigger already
      exists on the default branch — a brand-new trigger on a feature branch
      is not reliable on the PR's own checks.
- [ ] Merge to `main`. Confirm CI is green on `main` itself (a fresh `push`
      event on the real trigger config, not just the branch's).

## 5. Build verification

- [ ] Verify the built wheel in a clean venv:
      `pip install dist/*.whl && python -m yapss.examples.isoperimetric`

## 6. Tag & publish

- [ ] Tag `vX.Y.Z` and push. `hatch-vcs` derives the version from the tag.
- [ ] Publish to PyPI — prefer trusted publishing (OIDC) over a stored API
      token.

## 7. Conda

- [ ] Wait for the conda-forge autotick bot PR (hours, not immediate). It
      updates version and sha256 and resets the build number to 0, but does
      **not** sync dependency changes — apply those by hand if dependencies
      changed this release.
- [ ] Merge the bot PR from a fork. conda-forge rejects PRs from branches on
      the feedstock repo itself, even from maintainers.
- [ ] Verify: `conda create -n check -c conda-forge yapss`, then run the
      isoperimetric example.

## 8. Post-publish docs checks

- [ ] Confirm `stable` on Read the Docs resolves to the new version (RTD
      picks the highest semver tag by default, but check it wasn't manually
      pinned at some point).
- [ ] Spot-check the published RTD page itself, not just the source — a
      clean build doesn't guarantee the rendered output looks right.
- [ ] If the previous release had a known-broken install and a docs banner
      or notice was added for it, remove or update that notice now.
