"""Two layers of configuration.

**Server settings** — credentials and limits, from the environment. These
belong to whoever runs the bot.

**Repository settings** — `.reviewbot.yml`, committed to the repo being
reviewed. These belong to the team being reviewed, who are the people best
placed to say "ignore our generated migrations" or "we only want security
findings". Requiring a redeploy to change that would make the bot annoying to
live with.

Repository settings are clamped to the server's limits, so a repo cannot ask
for 500 comments and spam its own pull requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import Severity

log = logging.getLogger("prbot.config")

DEFAULT_IGNORE = [
    "*.lock",
    "*.min.js",
    "*.min.css",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "go.sum",
    "*.snap",
    "*.svg",
    "*.map",
    "dist/*",
    "build/*",
    "vendor/*",
    "node_modules/*",
    "*.generated.*",
    "*_pb2.py",
]


class Settings(BaseSettings):
    """Server configuration from the environment."""

    # --- credentials ---
    github_token: str = ""
    github_webhook_secret: str = ""
    anthropic_api_key: str = ""

    # --- model ---
    model: str = "claude-sonnet-4-5"
    max_tokens_per_request: int = 4096
    chunk_budget_tokens: int = 12_000

    # --- limits the repo config cannot exceed ---
    max_comments_hard_limit: int = 25
    max_file_lines: int = 1500
    max_concurrent_llm_calls: int = 4

    # --- defaults a repo may override ---
    default_max_comments: int = 10
    default_severity_threshold: Severity = Severity.MINOR

    # --- behaviour ---
    dry_run: bool = False          # analyse and log, never post
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def configured(self) -> bool:
        return bool(self.github_token and self.anthropic_api_key)

    def missing_credentials(self) -> list[str]:
        missing = []
        if not self.github_token:
            missing.append("GITHUB_TOKEN")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing


@dataclass
class RepoConfig:
    """Per-repository settings from `.reviewbot.yml`."""

    enabled: bool = True
    severity_threshold: Severity = Severity.MINOR
    max_comments: int = 10
    ignore_paths: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    focus: Optional[str] = None
    post_summary: bool = True

    @classmethod
    def default(cls, settings: Settings) -> "RepoConfig":
        return cls(
            severity_threshold=settings.default_severity_threshold,
            max_comments=settings.default_max_comments,
        )

    @classmethod
    def from_yaml(cls, raw: str, settings: Settings) -> "RepoConfig":
        """Parse `.reviewbot.yml`, falling back to defaults on anything invalid.

        A syntax error in a repo's config must not stop reviews — the author of
        the pull request usually didn't write that file and can't fix it. Log
        and carry on with defaults.
        """
        config = cls.default(settings)

        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            log.warning("invalid .reviewbot.yml, using defaults: %s", exc)
            return config

        if not isinstance(data, dict):
            log.warning(".reviewbot.yml is not a mapping, using defaults")
            return config

        if "enabled" in data:
            config.enabled = bool(data["enabled"])

        if "severity_threshold" in data:
            try:
                config.severity_threshold = Severity(str(data["severity_threshold"]).lower())
            except ValueError:
                log.warning(
                    "unknown severity_threshold %r, keeping %s",
                    data["severity_threshold"], config.severity_threshold.value,
                )

        if "max_comments" in data:
            try:
                requested = int(data["max_comments"])
                # Clamped: a repo cannot configure itself into being spammed.
                config.max_comments = max(1, min(requested, settings.max_comments_hard_limit))
            except (TypeError, ValueError):
                log.warning("invalid max_comments %r", data["max_comments"])

        if "ignore_paths" in data:
            patterns = data["ignore_paths"]
            if isinstance(patterns, list):
                # Extend rather than replace: the defaults exist because nobody
                # wants review comments on a lockfile, and silently dropping
                # them because a repo listed one extra pattern is surprising.
                config.ignore_paths = list(DEFAULT_IGNORE) + [str(p) for p in patterns]
            else:
                log.warning("ignore_paths must be a list")

        if "focus" in data and data["focus"]:
            config.focus = str(data["focus"])

        if "post_summary" in data:
            config.post_summary = bool(data["post_summary"])

        return config


settings = Settings()
