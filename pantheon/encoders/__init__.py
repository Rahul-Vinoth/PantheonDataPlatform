"""Importing this package registers all built-in encoders.

Add a new encoder by creating a module here with an `@register_encoder` decorator and
importing it below.
"""
from . import clip_encoder  # noqa: F401
