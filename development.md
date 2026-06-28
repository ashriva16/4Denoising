# Development Notes

This document is for contributors working on code, tests, packaging, or project
maintenance.

## Environment

Create the same user environment described in the README:

```bash
make env
source .venv/bin/activate
```

Install the project with development and test tools:

```bash
make dev
```

## Tests

Run the smoke-test suite with:

```bash
make test
```

or directly:

```bash
python -m pytest tests
```

The current tests use synthetic arrays only, so they do not need microscopy data,
checkpoints, or a GPU.

## Dependency Files

- `requirements.txt` is the user install file consumed by `make env`.
- `pyproject.toml` stores package metadata, optional development dependencies,
  and tool configuration.
- `environment.yaml` is kept as an optional Conda-style reference environment.

When adding a runtime dependency used by scripts or core modules, update the
user-facing install path as well as project metadata so fresh environments work.

## Developer References

- [Cookiecutter Docs](https://cookiecutter.readthedocs.io)
- [PEP 621](https://peps.python.org/pep-0621/)
- [GitHub Actions](https://docs.github.com/en/actions)
