"""GitHub API client.

Notable details
---------------
**Diffs are fetched with a media type, not a separate endpoint.** Asking for
``application/vnd.github.diff`` on the pull request URL returns the raw unified
diff. The alternative — the ``/files`` endpoint — returns per-file patches as
JSON, which is convenient until a file has been renamed or the patch is
omitted for being too large.

**Rate limits are respected rather than retried through.** GitHub returns the
reset time in a header; sleeping until then is correct, whereas backing off
blindly wastes the remaining quota on requests that will also fail.

**Idempotency is checked against existing comments.** Re-running on the same
commit must not double-post. The bot marks its own comments with a hidden HTML
marker and looks for it before posting.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import time
from hashlib import sha256
from typing import Optional

import httpx

from app.models import PlacedComment, PullRequestRef

log = logging.getLogger("prbot.github")

API = "https://api.github.com"

# Invisible in rendered markdown, but present in the comment body — how the bot
# recognises its own previous reviews.
BOT_MARKER = "<!-- pr-reviewer-bot -->"


class GitHubError(Exception):
    pass


def verify_signature(secret: str, body: bytes, signature: Optional[str]) -> bool:
    """Check a webhook's HMAC signature.

    Skipping this leaves an endpoint that anyone on the internet can use to
    make the bot review arbitrary repositories on your API budget. `compare_digest`
    rather than `==` because a plain comparison leaks, through timing, how much
    of a forged signature was correct.
    """
    if not secret:
        log.warning("no webhook secret configured — signature check skipped")
        return True
    if not signature or not signature.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(secret.encode(), body, sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class GitHubClient:
    def __init__(self, token: str, timeout: float = 30.0) -> None:
        self.token = token
        self._timeout = timeout

    def _headers(self, accept: str = "application/vnd.github+json") -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pr-reviewer-bot",
        }

    async def _request(
        self,
        method: str,
        url: str,
        accept: str = "application/vnd.github+json",
        **kwargs,
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            for attempt in range(3):
                response = await client.request(
                    method, url, headers=self._headers(accept), **kwargs
                )

                if response.status_code == 403 and _is_rate_limited(response):
                    wait = _seconds_until_reset(response)
                    log.warning("rate limited; sleeping %.0fs", wait)
                    await asyncio.sleep(min(wait, 60))
                    continue

                if response.status_code >= 500 and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue

                return response

        return response

    # ---------------------------------------------------------------- reads

    async def get_diff(self, pr: PullRequestRef) -> str:
        url = f"{API}/repos/{pr.full_name}/pulls/{pr.number}"
        response = await self._request("GET", url, accept="application/vnd.github.diff")

        if response.status_code != 200:
            raise GitHubError(
                f"could not fetch diff for {pr}: {response.status_code} {response.text[:200]}"
            )
        return response.text

    async def get_pull_request(self, pr: PullRequestRef) -> dict:
        url = f"{API}/repos/{pr.full_name}/pulls/{pr.number}"
        response = await self._request("GET", url)
        if response.status_code != 200:
            raise GitHubError(f"could not fetch {pr}: {response.status_code}")
        return response.json()

    async def get_repo_config(self, pr: PullRequestRef, path: str = ".reviewbot.yml") -> Optional[str]:
        """Read `.reviewbot.yml` from the PR's head branch.

        Read from the head, not the default branch, so a pull request that
        changes the bot's configuration is reviewed under the new settings —
        which is what someone editing that file expects.
        """
        url = f"{API}/repos/{pr.full_name}/contents/{path}"
        response = await self._request("GET", url, params={"ref": pr.head_sha})

        if response.status_code == 404:
            return None
        if response.status_code != 200:
            log.warning("could not read %s: %s", path, response.status_code)
            return None

        payload = response.json()
        if payload.get("encoding") != "base64":
            return None

        try:
            return base64.b64decode(payload["content"]).decode("utf-8")
        except Exception as exc:
            log.warning("could not decode %s: %s", path, exc)
            return None

    async def already_reviewed(self, pr: PullRequestRef) -> bool:
        """Has this bot already reviewed this exact commit?

        Guards against duplicate reviews when GitHub redelivers a webhook, or
        when a `synchronize` event fires without the head actually moving.
        """
        url = f"{API}/repos/{pr.full_name}/pulls/{pr.number}/reviews"
        response = await self._request("GET", url, params={"per_page": 100})

        if response.status_code != 200:
            return False          # fail open: a duplicate beats no review

        marker = _sha_marker(pr.head_sha)
        return any(marker in (review.get("body") or "") for review in response.json())

    # --------------------------------------------------------------- writes

    async def post_review(
        self,
        pr: PullRequestRef,
        comments: list[PlacedComment],
        summary: str,
        event: str = "COMMENT",
    ) -> dict:
        """Post inline comments and a summary as a single review.

        One review rather than N individual comments: it produces one
        notification instead of a flood, and it is atomic — either all comments
        land or none do, so a failure halfway through can't leave a partial
        review that a retry would then duplicate.
        """
        url = f"{API}/repos/{pr.full_name}/pulls/{pr.number}/reviews"
        body = {
            "commit_id": pr.head_sha,
            "body": f"{BOT_MARKER}\n{_sha_marker(pr.head_sha)}\n\n{summary}",
            "event": event,
            "comments": [c.to_github() for c in comments],
        }

        response = await self._request("POST", url, json=body)

        if response.status_code not in (200, 201):
            # A 422 here almost always means a comment pointed at a line not in
            # the diff. Surfacing the payload makes that debuggable instead of
            # mysterious.
            raise GitHubError(
                f"posting review failed: {response.status_code} {response.text[:400]}"
            )
        return response.json()

    async def post_issue_comment(self, pr: PullRequestRef, body: str) -> dict:
        """Fallback for when there are no inline comments to attach."""
        url = f"{API}/repos/{pr.full_name}/issues/{pr.number}/comments"
        response = await self._request(
            "POST", url, json={"body": f"{BOT_MARKER}\n{_sha_marker(pr.head_sha)}\n\n{body}"}
        )
        if response.status_code not in (200, 201):
            raise GitHubError(f"posting comment failed: {response.status_code}")
        return response.json()


def _sha_marker(sha: str) -> str:
    """Hidden marker recording which commit was reviewed."""
    return f"<!-- reviewed-sha: {sha} -->"


def _is_rate_limited(response: httpx.Response) -> bool:
    return response.headers.get("x-ratelimit-remaining") == "0"


def _seconds_until_reset(response: httpx.Response) -> float:
    reset = response.headers.get("x-ratelimit-reset")
    if not reset:
        return 60.0
    try:
        return max(1.0, float(reset) - time.time())
    except ValueError:
        return 60.0
