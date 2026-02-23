"""
helix/safety/permissions.py

Per-agent permission scoping, enforced at call time.

Law 7: Security by Scope, Not by Policy Document.
No agent can exceed its declared capability surface.
"""

from __future__ import annotations

from helix.config import PermissionConfig
from helix.errors import PermissionDeniedError


class PermissionScope:
    """
    Enforces tool and resource access for a single agent.

    Built from PermissionConfig at agent initialization.
    All checks happen at call time, not at deployment time.
    """

    def __init__(self, config: PermissionConfig, agent_id: str = "") -> None:
        self._agent_id = agent_id
        self._allowed_tools: set[str] | None = (
            set(config.allowed_tools) if config.allowed_tools is not None else None
        )
        self._denied_tools: set[str] = set(config.denied_tools)
        self._allowed_domains: set[str] | None = (
            set(config.allowed_domains) if config.allowed_domains is not None else None
        )
        self._max_file_size_mb = config.max_file_size_mb

    def check_tool(self, tool_name: str) -> None:
        """Raise PermissionDeniedError if tool is not permitted."""
        if tool_name in self._denied_tools:
            raise PermissionDeniedError(
                agent_id=self._agent_id,
                action="call_tool",
                resource=tool_name,
            )
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            raise PermissionDeniedError(
                agent_id=self._agent_id,
                action="call_tool",
                resource=tool_name,
            )

    def check_domain(self, domain: str) -> None:
        """Raise PermissionDeniedError if domain is not in allowed list."""
        if self._allowed_domains is None:
            return  # No restriction
        if domain not in self._allowed_domains:
            raise PermissionDeniedError(
                agent_id=self._agent_id,
                action="access_domain",
                resource=domain,
            )

    def check_file_size(self, size_mb: float) -> None:
        if size_mb > self._max_file_size_mb:
            raise PermissionDeniedError(
                agent_id=self._agent_id,
                action="read_file",
                resource=f"{size_mb:.1f}MB file (limit: {self._max_file_size_mb}MB)",
            )

    def is_tool_permitted(self, tool_name: str) -> bool:
        try:
            self.check_tool(tool_name)
            return True
        except PermissionDeniedError:
            return False

    def permitted_tools(self, all_tools: list[str]) -> list[str]:
        return [t for t in all_tools if self.is_tool_permitted(t)]


def allow(*tool_names: str) -> PermissionConfig:
    """Convenience factory: create a PermissionConfig that allows only named tools."""
    from helix.config import PermissionConfig

    return PermissionConfig(allowed_tools=list(tool_names))


def deny(*tool_names: str) -> PermissionConfig:
    """Convenience factory: create a PermissionConfig that denies named tools."""
    from helix.config import PermissionConfig

    return PermissionConfig(denied_tools=list(tool_names))
