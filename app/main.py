"""FastAPI webhook receiver.

The timing constraint
---------------------
GitHub expects a webhook response within about 10 seconds and marks the
delivery failed otherwise. A review takes far longer — fetching the diff,
several LLM calls, posting results. So the handler verifies the signature,
acknowledges immediately, and does the work in the background.

Getting this backwards is the single most common webhook mistake: GitHub times
out, retries, and now two reviews run concurrently on the same commit.
(The idempotency check in `GitHubClient.already_reviewed` is the second line of
defence against exactly that.)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from app.config import settings
from app.github import GitHubClient, verify_signature
from app.llm import ClaudeClient
from app.models import PullRequestRef
from app.review import Reviewer

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prbot")

# Events worth acting on. `opened` is the first review; `synchronize` fires when
# new commits are pushed, which needs a fresh review of the new head.
REVIEWABLE_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = settings.missing_credentials()
    if missing:
        log.warning(
            "starting without %s — /webhook will accept and log events but "
            "cannot review. Set them in .env to enable reviewing.",
            ", ".join(missing),
        )
    if settings.dry_run:
        log.info("DRY RUN: reviews will be computed and logged, never posted")
    yield


app = FastAPI(
    title="PR Review Bot",
    description="Reviews pull requests with Claude and posts inline comments.",
    version="1.0.0",
    lifespan=lifespan,
)


def build_reviewer() -> Reviewer:
    return Reviewer(
        github=GitHubClient(settings.github_token),
        llm=ClaudeClient(settings.anthropic_api_key, model=settings.model),
        settings=settings,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "configured": settings.configured,
        "missing": settings.missing_credentials(),
        "dry_run": settings.dry_run,
        "model": settings.model,
    }


@app.post("/webhook")
async def webhook(
    request: Request,
    background: BackgroundTasks,
    x_github_event: str = Header(default=""),
    x_hub_signature_256: str = Header(default=""),
):
    body = await request.body()

    # Verified before parsing: an unverified endpoint lets anyone on the
    # internet make this bot review arbitrary repos on your API budget.
    if not verify_signature(settings.github_webhook_secret, body, x_hub_signature_256):
        log.warning("rejected webhook with invalid signature")
        raise HTTPException(status_code=401, detail="invalid signature")

    if x_github_event == "ping":
        return {"status": "pong"}

    if x_github_event != "pull_request":
        return {"status": "ignored", "reason": f"event '{x_github_event}' not handled"}

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")

    action = payload.get("action", "")
    if action not in REVIEWABLE_ACTIONS:
        return {"status": "ignored", "reason": f"action '{action}' not reviewable"}

    pull = payload.get("pull_request") or {}
    if pull.get("draft"):
        return {"status": "ignored", "reason": "pull request is a draft"}

    try:
        pr = PullRequestRef(
            owner=payload["repository"]["owner"]["login"],
            repo=payload["repository"]["name"],
            number=pull["number"],
            head_sha=pull["head"]["sha"],
        )
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"malformed payload: {exc}")

    if not settings.configured:
        log.warning("received %s for %s but credentials are missing", action, pr)
        return {"status": "accepted", "reviewing": False, "reason": "not configured"}

    # Acknowledge now, review afterwards.
    background.add_task(_run_review, pr)
    log.info("accepted %s for %s", action, pr)

    return {"status": "accepted", "reviewing": True, "pr": str(pr)}


async def _run_review(pr: PullRequestRef) -> None:
    """Background entry point. Must never raise — nothing is listening."""
    try:
        result = await build_reviewer().review_pull_request(pr)
        log.info(result.summary_line())
    except Exception:
        log.exception("review failed for %s", pr)


@app.post("/review/{owner}/{repo}/{number}")
async def review_manually(owner: str, repo: str, number: int):
    """Trigger a review by hand, without a webhook.

    Makes the bot demonstrable on an existing pull request without configuring
    webhook delivery — which is how you'd want to try it the first time.
    """
    if not settings.configured:
        raise HTTPException(
            status_code=503,
            detail=f"missing credentials: {', '.join(settings.missing_credentials())}",
        )

    github = GitHubClient(settings.github_token)
    ref = PullRequestRef(owner=owner, repo=repo, number=number, head_sha="")

    data = await github.get_pull_request(ref)
    ref.head_sha = data["head"]["sha"]

    result = await build_reviewer().review_pull_request(ref)
    return {
        "pr": str(ref),
        "files_reviewed": result.files_reviewed,
        "findings_raw": result.findings_raw,
        "comments_posted": result.comments_posted,
        "estimated_cost_usd": round(result.estimated_cost_usd, 4),
        "skipped_reason": result.skipped_reason,
    }
