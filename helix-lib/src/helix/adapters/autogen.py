"""
helix/adapters/autogen.py

AutoGen-specific adapter.
Re-exports from universal.py for clean import paths.

Usage::

    from helix.adapters.autogen import from_autogen
"""

from helix.adapters.universal import AutoGenWrapper, from_autogen

__all__ = ["from_autogen", "AutoGenWrapper"]
