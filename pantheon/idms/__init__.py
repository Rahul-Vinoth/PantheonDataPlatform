"""Importing this package registers all built-in IDMs.

Add a new IDM by creating a module here with an `@register_idm` decorator and importing
it below.
"""
from . import delta_idm  # noqa: F401
