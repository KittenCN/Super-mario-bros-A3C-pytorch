"""Top-level package marker to help mypy unify module paths.

Having this file allows mypy to treat the repository root as a package root,
preventing duplicate module detection between `config` and `src.config`.
"""
