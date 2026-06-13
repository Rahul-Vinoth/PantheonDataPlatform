"""Importing this package registers all built-in adapters (Trossen #1).

Add a new adapter by creating a module here with an `@register_adapter` decorator and
importing it below.
"""
from . import ego_raw  # noqa: F401
from . import egodex   # noqa: F401
from . import lerobot  # noqa: F401
