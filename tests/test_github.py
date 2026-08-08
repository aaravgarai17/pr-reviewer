"""GitHub client and webhook signature verification, with HTTP mocked."""

import base64
import hmac
from hashlib import sha256

import httpx
import pytest
import respx

from app.github import BOT_MARKER, GitHubClient, GitHubError, verify_signature
from app.models import PlacedComment, PullRequestRef, Severity

API = "https://api.github.com"


@pytest.fixture
def pr():
    return PullRequestRef(owner="octocat", repo="hello", number=7, head_sha="abc1234def")


@pytest.fixture
def client():
    return GitHubClient(token="ghp_test")


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, sha256).hexdigest()


# ------------------------------------------------------- signature verification


def test_valid_signature_accepted():
    body = b'{"action":"opened"}'
    assert verify_signature("s3cret", body, sign("s3cret", body)) is True


def test_wrong_secret_rejected():
    body = b'{"action":"opened"}'
    assert verify_signature("s3cret", body, sign("wrong", body)) is False


def test_tampered_body_rejected():
    """The whole point: a modified payload must not verify."""
    signature = sign("s3cret", b'{"action":"opened"}')
    assert verify_signature("s3cret", b'{"action":"evil"}', signature) is False


def test_missing_signature_rejected():
    assert verify_signature("s3cret", b"{}", None) is False
    assert verify_signature("s3cret", b"{}", "") is False


def test_wrong_prefix_rejected():
    body = b"{}"
    digest = hmac.new(b"s3cret", body, sha256).hexdigest()
    assert verify_signature("s3cret", body, f"sha1={digest}") is False


def test_no_secret_configured_skips_verification():
    """Documented escape hatch for local development."""
    assert verify_signature("", b"{}", None) is True


# ----------------------------------------------------------------- fetching


@respx.mock
async def test_get_diff(client, pr):
    respx.get(f"{API}/repos/octocat/hello/pulls/7").mock(
        return_value=httpx.Response(200, text="diff --git a/x.py b/x.py\n")
    )
    assert "diff --git" in await client.get_diff(pr)


@respx.mock
async def test_get_diff_requests_the_diff_media_type(client, pr):
    route = respx.get(f"{API}/repos/octocat/hello/pulls/7").mock(
        return_value=httpx.Response(200, text="")
    )
    await client.get_diff(pr)
    assert route.calls[0].request.headers["accept"] == "application/vnd.github.diff"


@respx.mock
async def test_get_diff_raises_on_failure(client, pr):
    respx.get(f"{API}/repos/octocat/hello/pulls/7").mock(
        return_value=httpx.Response(404, text="Not Found")
    )
    with pytest.raises(GitHubError):
        await client.get_diff(pr)


@respx.mock
async def test_get_repo_config_decodes_base64(client, pr):
    content = base64.b64encode(b"max_comments: 3").decode()
    respx.get(f"{API}/repos/octocat/hello/contents/.reviewbot.yml").mock(
        return_value=httpx.Response(200, json={"encoding": "base64", "content": content})
    )
    assert await client.get_repo_config(pr) == "max_comments: 3"


@respx.mock
async def test_missing_repo_config_returns_none(client, pr):
    respx.get(f"{API}/repos/octocat/hello/contents/.reviewbot.yml").mock(
        return_value=httpx.Response(404)
    )
    assert await client.get_repo_config(pr) is None


@respx.mock
async def test_repo_config_read_from_the_head_sha(client, pr):
    """So a PR that edits .reviewbot.yml is reviewed under its new settings."""
    route = respx.get(f"{API}/repos/octocat/hello/contents/.reviewbot.yml").mock(
        return_value=httpx.Response(404)
    )
    await client.get_repo_config(pr)
    assert route.calls[0].request.url.params["ref"] == "abc1234def"


# -------------------------------------------------------------- idempotency


@respx.mock
async def test_already_reviewed_detects_the_sha_marker(client, pr):
    respx.get(f"{API}/repos/octocat/hello/pulls/7/reviews").mock(
        return_value=httpx.Response(
            200,
            json=[{"body": f"{BOT_MARKER}\n<!-- reviewed-sha: abc1234def -->\n\nLGTM"}],
        )
    )
    assert await client.already_reviewed(pr) is True


@respx.mock
async def test_review_of_a_different_commit_does_not_count(client, pr):
    """New commits must get a fresh review."""
    respx.get(f"{API}/repos/octocat/hello/pulls/7/reviews").mock(
        return_value=httpx.Response(
            200, json=[{"body": f"{BOT_MARKER}\n<!-- reviewed-sha: 999999 -->"}]
        )
    )
    assert await client.already_reviewed(pr) is False


@respx.mock
async def test_human_reviews_do_not_count(client, pr):
    respx.get(f"{API}/repos/octocat/hello/pulls/7/reviews").mock(
        return_value=httpx.Response(200, json=[{"body": "looks good to me"}])
    )
    assert await client.already_reviewed(pr) is False


@respx.mock
async def test_idempotency_check_fails_open(client, pr):
    """A duplicate review beats no review when the check itself errors."""
    respx.get(f"{API}/repos/octocat/hello/pulls/7/reviews").mock(
        return_value=httpx.Response(500)
    )
    assert await client.already_reviewed(pr) is False


# ------------------------------------------------------------------ posting


@respx.mock
async def test_post_review_sends_one_payload(client, pr):
    route = respx.post(f"{API}/repos/octocat/hello/pulls/7/reviews").mock(
        return_value=httpx.Response(200, json={"id": 1})
    )
    comments = [
        PlacedComment(
            path="a.py", line=5, position=3, body="Possible None.", severity=Severity.MAJOR
        )
    ]
    await client.post_review(pr, comments, "Found 1 issue.")

    body = route.calls[0].request.content.decode()
    assert '"commit_id":"abc1234def"' in body.replace(" ", "")
    assert '"path":"a.py"' in body.replace(" ", "")
    assert '"line":5' in body.replace(" ", "")


@respx.mock
async def test_post_review_embeds_the_sha_marker(client, pr):
    """The marker is what makes re-runs idempotent."""
    route = respx.post(f"{API}/repos/octocat/hello/pulls/7/reviews").mock(
        return_value=httpx.Response(200, json={})
    )
    await client.post_review(pr, [], "summary")

    body = route.calls[0].request.content.decode()
    assert "reviewed-sha: abc1234def" in body
    assert BOT_MARKER in body


@respx.mock
async def test_post_review_surfaces_422_detail(client, pr):
    """422 almost always means a comment pointed outside the diff.

    Including the response body makes that debuggable rather than mysterious.
    """
    respx.post(f"{API}/repos/octocat/hello/pulls/7/reviews").mock(
        return_value=httpx.Response(422, text='{"message":"line must be part of the diff"}')
    )
    with pytest.raises(GitHubError, match="line must be part of the diff"):
        await client.post_review(pr, [], "summary")


@respx.mock
async def test_post_issue_comment(client, pr):
    route = respx.post(f"{API}/repos/octocat/hello/issues/7/comments").mock(
        return_value=httpx.Response(201, json={})
    )
    await client.post_issue_comment(pr, "Nothing to flag.")
    assert "Nothing to flag." in route.calls[0].request.content.decode()


# -------------------------------------------------------------- resilience


@respx.mock
async def test_retries_on_server_error(client, pr):
    route = respx.get(f"{API}/repos/octocat/hello/pulls/7").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, text="diff --git a/x b/x\n"),
        ]
    )
    assert "diff --git" in await client.get_diff(pr)
    assert route.call_count == 2
