"""
helix/adapters/crewai.py

CrewAI-specific adapter.
Re-exports from universal.py for clean import paths.

Usage::

    from helix.adapters.crewai import from_crewai
"""

from helix.adapters.universal import from_crewai, CrewAIWrapper

__all__ = ["from_crewai", "CrewAIWrapper"]
