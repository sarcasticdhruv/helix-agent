"""
helix/memory/wal.py

Write-ahead log — re-exported from store for direct import.
The implementation lives in MemoryStore.WriteAheadLog.
This module exposes it standalone for testing and external use.
"""

from helix.memory.store import WriteAheadLog

__all__ = ["WriteAheadLog"]
