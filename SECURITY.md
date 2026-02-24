# Security Policy

## Supported versions

| Version | Security fixes |
|---------|:--------------:|
| 0.3.x   | ✅             |
| 0.2.x   | ✅             |
| < 0.2   | ❌             |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately by emailing the maintainer:

- **Dhruv Choudhary** — use GitHub's
  [private vulnerability reporting](../../security/advisories/new) feature
  (recommended), or email via the address on the GitHub profile.

Please include:

1. A description of the vulnerability and its potential impact.
2. Steps to reproduce or a proof-of-concept.
3. Affected versions.
4. Any suggested remediation (optional).

**Response SLA:** You will receive an acknowledgement within **48 hours** and a
resolution or status update within **7 days**.

## Security considerations for users

- **API keys** — Helix stores provider keys in `~/.helix/config.json`. Ensure
  this file has restricted permissions (`chmod 600 ~/.helix/config.json` on
  Unix-like systems).
- **Tool permissions** — use `PermissionConfig(denied_tools=[...])` to restrict
  which tools agents can invoke in production deployments.
- **Audit log** — enable `helix.AuditConfig` to log all LLM calls and tool
  invocations for compliance purposes.
- **Budget enforcement** — always set `AgentMode.PRODUCTION` with a
  `BudgetConfig` to prevent runaway API spend.
