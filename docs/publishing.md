# Publishing to PyPI

The package name `cx-tda-engine` (import name `cx_tda_engine`) was free on
PyPI and TestPyPI on 2026-10-08. Nothing has been uploaded yet.

## Pre-flight (already in place)

* `pyproject.toml`: PEP 621 metadata, MIT license + `LICENSE` file, classifiers,
  project URLs, `py.typed`, README as long description.
* `python -m build` produces `dist/cx_tda_engine-<version>-py3-none-any.whl` and the sdist;
  `twine check --strict dist/*` passes.
* `.github/workflows/publish.yml`: manual workflow (Actions tab -> publish -> Run) that builds,
  checks and uploads to TestPyPI or PyPI through trusted publishing. It never runs on its own.

## One-time account setup

1. Create accounts on https://test.pypi.org and https://pypi.org, enable 2FA.
2. Trusted publishing (recommended, no token in GitHub secrets): on each site go to
   *Your projects -> Publishing -> Add a new pending publisher* with
   project `cx-tda-engine`, owner `NarayanYerrabachu`, repository `cx_tda_engine`,
   workflow `publish.yml`, environment `testpypi` (on TestPyPI) / `pypi` (on PyPI).
   In GitHub, create the two environments under *Settings -> Environments*; put a
   required reviewer on `pypi` so a release needs an explicit approval click.
3. Alternative, from your laptop: create an API token on PyPI and keep it in `~/.pypirc`
   (never in the repo).

## Release steps

```bash
# 1. bump the version in pyproject.toml, add a CHANGELOG entry, commit
# 2. tag and push (release.yml attaches the wheel to the GitHub release)
git tag v0.1.0 && git push origin main v0.1.0
# 3a. GitHub: Actions -> publish -> Run workflow -> target testpypi, verify, then target pypi
# 3b. or locally:
export PIPENV_CUSTOM_VENV_NAME=cx_tda_engine
pipenv run python -m build
pipenv run twine check --strict dist/*
pipenv run twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cx-tda-engine
pipenv run twine upload dist/*
```

After the first upload, users install with:

```bash
pip install cx-tda-engine
pip install "cx-tda-engine[viz]"
```

A version can never be re-uploaded to PyPI; a fix means a new patch version.
