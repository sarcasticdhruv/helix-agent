"""
helix/safety/audit.py

Append-only, tamper-evident audit log.

Every entry includes a hash of the previous entry, forming a chain.
The chain can be verified to detect tampering or log truncation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import AsyncIterator, Dict, Optional, Tuple

from helix.config import AuditEntry, AuditEventType
from helix.interfaces import AuditLogBackend


def _hash_entry(entry: AuditEntry) -> str:
    """Produce a deterministic hash of an audit entry's content."""
    canonical = json.dumps({
        "id": entry.id,
        "event_type": entry.event_type.value,
        "agent_id": entry.agent_id,
        "details": entry.details,
        "timestamp": entry.timestamp,
        "prev_hash": entry.prev_hash,
    }, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class LocalFileAuditLog(AuditLogBackend):
    """
    Writes audit entries as JSON lines to a local file.

    Each entry includes prev_hash (hash of previous entry) and
    entry_hash (hash of this entry's content) to form a verifiable chain.

    Not suitable for multi-process / multi-replica deployments.
    Use S3AuditBackend for production multi-instance deployments.
    """

    def __init__(self, agent_id: str, log_dir: str = ".helix/audit") -> None:
        self._agent_id = agent_id
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._dir / f"{agent_id}.jsonl"
        self._last_hash: str = ""
        self._lock = asyncio.Lock()
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Restore the last hash from disk to continue the chain."""
        if not self._log_path.exists():
            self._last_hash = ""
            return
        last_line = ""
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    last_line = line.strip()
        except Exception:
            return
        if last_line:
            try:
                data = json.loads(last_line)
                self._last_hash = data.get("entry_hash", "")
            except Exception:
                self._last_hash = ""

    async def append(self, entry: AuditEntry) -> None:
        async with self._lock:
            # Build hash chain
            entry_with_prev = entry.model_copy(update={"prev_hash": self._last_hash})
            entry_hash = _hash_entry(entry_with_prev)
            final_entry = entry_with_prev.model_copy(update={"entry_hash": entry_hash})

            line = json.dumps(final_entry.model_dump(mode="json"), default=str) + "\n"
            try:
                with open(self._log_path, "a") as f:
                    f.write(line)
                self._last_hash = entry_hash
            except OSError:
                pass  # Non-fatal — audit failure should not crash the agent

    async def export(
        self,
        since_timestamp: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> AsyncIterator[AuditEntry]:
        if not self._log_path.exists():
            return

        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                    if since_timestamp and entry.timestamp < since_timestamp:
                        continue
                    if agent_id and entry.agent_id != agent_id:
                        continue
                    yield entry
                except Exception:
                    continue

    async def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """Walk the log file and verify each entry's prev_hash matches."""
        if not self._log_path.exists():
            return True, None

        prev_hash = ""
        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                    if entry.prev_hash != prev_hash:
                        return False, entry.id
                    # Recompute expected hash
                    expected = _hash_entry(entry)
                    if entry.entry_hash != expected:
                        return False, entry.id
                    prev_hash = entry.entry_hash
                except Exception as e:
                    return False, str(e)

        return True, None

    async def health(self) -> bool:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
