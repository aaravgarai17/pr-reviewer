"""Repository configuration parsing and clamping."""

import pytest

from app.config import DEFAULT_IGNORE, RepoConfig, Settings
from app.models import Severity


@pytest.fixture
def settings():
    return Settings(
        max_comments_hard_limit=25,
        default_max_comments=10,
        default_severity_threshold=Severity.MINOR,
    )


def test_defaults_when_no_file(settings):
    config = RepoConfig.default(settings)
    assert config.enabled is True
    assert config.max_comments == 10
    assert config.severity_threshold is Severity.MINOR


def test_parses_a_full_config(settings):
    yaml_text = """
enabled: true
severity_threshold: major
max_comments: 5
focus: security and error handling
post_summary: false
ignore_paths:
  - "migrations/*"
"""
    config = RepoConfig.from_yaml(yaml_text, settings)

    assert config.severity_threshold is Severity.MAJOR
    assert config.max_comments == 5
    assert config.focus == "security and error handling"
    assert config.post_summary is False
    assert "migrations/*" in config.ignore_paths


def test_ignore_paths_extend_rather_than_replace_defaults(settings):
    """Listing one pattern shouldn't silently re-enable lockfile reviews."""
    config = RepoConfig.from_yaml('ignore_paths:\n  - "custom/*"', settings)

    assert "custom/*" in config.ignore_paths
    assert "package-lock.json" in config.ignore_paths
    assert all(p in config.ignore_paths for p in DEFAULT_IGNORE)


def test_max_comments_is_clamped_to_the_server_limit(settings):
    """A repo must not be able to configure itself into being spammed."""
    config = RepoConfig.from_yaml("max_comments: 500", settings)
    assert config.max_comments == 25


def test_max_comments_has_a_floor(settings):
    config = RepoConfig.from_yaml("max_comments: 0", settings)
    assert config.max_comments == 1


def test_disabled_repo(settings):
    assert RepoConfig.from_yaml("enabled: false", settings).enabled is False


def test_invalid_yaml_falls_back_to_defaults(settings):
    """A broken config must not block reviews.

    The pull request author usually didn't write that file and can't fix it.
    """
    config = RepoConfig.from_yaml("this: [is: not: valid: yaml", settings)
    assert config.max_comments == 10
    assert config.enabled is True


def test_non_mapping_yaml_falls_back(settings):
    config = RepoConfig.from_yaml("- just\n- a\n- list", settings)
    assert config.max_comments == 10


def test_unknown_severity_keeps_the_default(settings):
    config = RepoConfig.from_yaml("severity_threshold: catastrophic", settings)
    assert config.severity_threshold is Severity.MINOR


def test_non_numeric_max_comments_keeps_the_default(settings):
    config = RepoConfig.from_yaml("max_comments: many", settings)
    assert config.max_comments == 10


def test_ignore_paths_must_be_a_list(settings):
    config = RepoConfig.from_yaml('ignore_paths: "just-a-string"', settings)
    assert config.ignore_paths == DEFAULT_IGNORE


def test_empty_file_gives_defaults(settings):
    config = RepoConfig.from_yaml("", settings)
    assert config.max_comments == 10


# ------------------------------------------------------------------ settings


def test_missing_credentials_reported():
    s = Settings(github_token="", anthropic_api_key="")
    assert set(s.missing_credentials()) == {"GITHUB_TOKEN", "ANTHROPIC_API_KEY"}
    assert s.configured is False


def test_configured_when_both_present():
    s = Settings(github_token="ghp_x", anthropic_api_key="sk-ant-x")
    assert s.configured is True
    assert s.missing_credentials() == []
