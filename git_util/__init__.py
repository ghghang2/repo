"""Utility package for GitHub interactions.

The original repository omitted ``__init__`` which caused an import error
when the ``push_to_github`` helper was executed from the repository root.
Adding this file makes ``git_util`` a proper Python package and restores
the original behaviour of the helper scripts.
"""