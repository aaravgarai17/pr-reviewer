"""Webhook endpoint: signature enforcement, event filtering, fast acknowledgement."""

import hmac
import json
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app

SECRET = "test-secret"


@pytest.fixture(autouse=True)
def configure(monkeypatch):
    monkeypatch.setattr(main.settings, "github_webhook_secret", SECRET)
    monkeypatch.setattr(main.settings, "github_token", "ghp_test")
    monkeypatch.setattr(main.settings, "anthropic_api_key", "sk-ant-test")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def never_actually_review(monkeypatch):
    """Capture background review calls instead of running them."""
    called = []

    async def fake(pr):
        called.append(pr)

    monkeypatch.setattr(main, "_run_review", fake)
    return called


def payload(action="opened", draft=False, sha="abc123"):
    return {
        "action": action,
        "repository": {"owner": {"login": "octocat"}, "name": "hello"},
        "pull_request": {"number": 7, "draft": draft, "head": {"sha": sha}},
    }


def post(client, body: dict, event="pull_request", secret=SECRET):
    raw = json.dumps(body).encode()
    signature = "sha256=" + hmac.new(secret.encode(), raw, sha256).hexdigest()
    return client.post(
        "/webhook",
        content=raw,
        headers={
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": signature,
            "Content-Type": "application/json",
        },
    )


# ------------------------------------------------------------------- health


def test_health_reports_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["configured"] is True


# ----------------------------------------------------------------- security


def test_valid_signature_accepted(client):
    assert post(client, payload()).status_code == 200


def test_invalid_signature_rejected(client):
    """An unverified endpoint lets anyone spend your API budget."""
    response = post(client, payload(), secret="wrong-secret")
    assert response.status_code == 401


def test_missing_signature_rejected(client):
    response = client.post(
        "/webhook",
        content=json.dumps(payload()).encode(),
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert response.status_code == 401


def test_tampered_body_rejected(client):
    """Signature covers the body, so modifying it must invalidate the request."""
    raw = json.dumps(payload()).encode()
    signature = "sha256=" + hmac.new(SECRET.encode(), raw, sha256).hexdigest()

    tampered = json.dumps(payload(sha="evil")).encode()
    response = client.post(
        "/webhook",
        content=tampered,
        headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": signature},
    )
    assert response.status_code == 401


# ------------------------------------------------------------ event routing


def test_ping_event(client):
    assert post(client, {"zen": "hi"}, event="ping").json()["status"] == "pong"


def test_unrelated_events_ignored(client, never_actually_review):
    response = post(client, {"ref": "main"}, event="push")
    assert response.json()["status"] == "ignored"
    assert never_actually_review == []


@pytest.mark.parametrize("action", ["opened", "synchronize", "reopened", "ready_for_review"])
def test_reviewable_actions_accepted(client, never_actually_review, action):
    response = post(client, payload(action=action))
    assert response.json()["reviewing"] is True
    assert len(never_actually_review) == 1


@pytest.mark.parametrize("action", ["closed", "labeled", "assigned", "edited"])
def test_non_reviewable_actions_ignored(client, never_actually_review, action):
    response = post(client, payload(action=action))
    assert response.json()["status"] == "ignored"
    assert never_actually_review == []


def test_draft_pull_requests_are_ignored(client, never_actually_review):
    """Work in progress; reviewing it would be premature and noisy."""
    response = post(client, payload(draft=True))
    assert response.json()["status"] == "ignored"
    assert "draft" in response.json()["reason"]
    assert never_actually_review == []


# ------------------------------------------------------------- acknowledgement


def test_review_runs_in_the_background(client, never_actually_review):
    """GitHub times out after ~10s; the work must not happen inline.

    Responding late causes a retry, and now two reviews race on one commit.
    """
    response = post(client, payload())

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert len(never_actually_review) == 1

    pr = never_actually_review[0]
    assert pr.owner == "octocat"
    assert pr.repo == "hello"
    assert pr.number == 7
    assert pr.head_sha == "abc123"


def test_malformed_payload_rejected(client):
    response = post(client, {"action": "opened", "repository": {}})
    assert response.status_code == 400


def test_accepts_but_does_not_review_without_credentials(client, monkeypatch, never_actually_review):
    """Returning 200 keeps GitHub from retrying a delivery we can't act on."""
    monkeypatch.setattr(main.settings, "github_token", "")
    monkeypatch.setattr(main.settings, "anthropic_api_key", "")

    response = post(client, payload())
    assert response.status_code == 200
    assert response.json()["reviewing"] is False
    assert never_actually_review == []
