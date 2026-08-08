"""The review pipeline.

Fetch → parse → filter → chunk → review → anchor → filter → post.

Deliberately depends on `GitHubClient` and `LLMClient` by interface rather than
constructing them, so the whole pipeline runs in tests against stubs — no API
keys, no network, no cost. Only the two clients touch the outside world.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Optional

from app.chunking import filter_reviewable, plan_chunks
from app.config import RepoConfig, Settings
from app.diff import parse_diff
from app.filters import filter_pipeline
from app.github import GitHubClient
from app.llm import LLMClient, request_findings
from app.models import PullRequestRef, ReviewResult, Severity
from app.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_review_prompt,
    build_summary_prompt,
)

log = logging.getLogger("prbot.review")


class Reviewer:
    def __init__(
        self,
        github: GitHubClient,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self.github = github
        self.llm = llm
        self.settings = settings

    async def review_pull_request(self, pr: PullRequestRef) -> ReviewResult:
        result = ReviewResult(pr=pr)

        # --- idempotency -----------------------------------------------------
        if await self.github.already_reviewed(pr):
            log.info("%s already reviewed at this commit, skipping", pr)
            result.skipped_reason = "already reviewed at this commit"
            return result

        # --- per-repo configuration -----------------------------------------
        raw_config = await self.github.get_repo_config(pr)
        config = (
            RepoConfig.from_yaml(raw_config, self.settings)
            if raw_config
            else RepoConfig.default(self.settings)
        )

        if not config.enabled:
            log.info("%s: reviews disabled by .reviewbot.yml", pr)
            result.skipped_reason = "disabled by .reviewbot.yml"
            return result

        # --- fetch and parse -------------------------------------------------
        diff_text = await self.github.get_diff(pr)
        files = parse_diff(diff_text)

        reviewable, skipped = filter_reviewable(
            files,
            ignore_patterns=config.ignore_paths,
            max_file_lines=self.settings.max_file_lines,
        )
        result.files_reviewed = len(reviewable)
        result.files_skipped = len(skipped)

        for path, reason in skipped:
            log.debug("skipping %s: %s", path, reason)

        if not reviewable:
            log.info("%s: nothing reviewable", pr)
            result.skipped_reason = "no reviewable files"
            if config.post_summary:
                await self.github.post_issue_comment(
                    pr, "No reviewable code changes found in this pull request."
                )
            return result

        # --- chunk and review ------------------------------------------------
        plan = plan_chunks(
            reviewable,
            budget_tokens=self.settings.chunk_budget_tokens,
        )
        result.chunks_sent = len(plan.chunks)
        log.info("%s: %d file(s) in %d chunk(s)", pr, len(reviewable), len(plan.chunks))

        responses = await self._review_chunks(plan.chunks, config.focus)

        findings = []
        for response in responses:
            findings.extend(response.findings)
            result.input_tokens += response.input_tokens
            result.output_tokens += response.output_tokens

        result.findings_raw = len(findings)

        # --- noise control ---------------------------------------------------
        files_by_path = {f.path: f for f in reviewable}
        comments, stats = filter_pipeline(
            findings,
            files_by_path,
            threshold=config.severity_threshold,
            max_comments=config.max_comments,
        )
        result.findings_kept = len(comments)

        log.info(
            "%s: %d raw → %d kept (%d unanchorable, %d below threshold, "
            "%d duplicates, %d over cap)",
            pr, stats["raw"], stats["kept"], stats["unanchorable"],
            stats["below_threshold"], stats["duplicates"], stats["over_cap"],
        )

        # --- post ------------------------------------------------------------
        summary = self._build_summary(comments, result, stats)

        if self.settings.dry_run:
            log.info("[dry run] would post %d comment(s)", len(comments))
            for c in comments:
                log.info("  %s:%s [%s] %s", c.path, c.line, c.severity.value, c.body[:80])
            result.comments_posted = 0
            return result

        if comments:
            await self.github.post_review(pr, comments, summary)
            result.comments_posted = len(comments)
        elif config.post_summary:
            await self.github.post_issue_comment(pr, summary)

        log.info(result.summary_line())
        return result

    async def _review_chunks(self, chunks, focus: Optional[str]):
        """Review chunks concurrently, bounded.

        Unbounded concurrency on a 50-file PR would fire 50 simultaneous
        requests and hit the provider's rate limit, turning a slow review into
        a failed one. A semaphore keeps it to a few at a time.
        """
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_llm_calls)

        async def review_one(chunk):
            async with semaphore:
                prompt = build_review_prompt(chunk.rendered, focus)
                return await request_findings(
                    self.llm,
                    SYSTEM_PROMPT,
                    prompt,
                    max_tokens=self.settings.max_tokens_per_request,
                )

        results = await asyncio.gather(
            *(review_one(c) for c in chunks), return_exceptions=True
        )

        good = []
        for item in results:
            if isinstance(item, Exception):
                # One failed chunk shouldn't lose the whole review — post what
                # the other chunks found.
                log.error("chunk failed: %s", item)
                continue
            good.append(item)
        return good

    def _build_summary(self, comments, result: ReviewResult, stats: dict) -> str:
        """Compose the review summary without spending another LLM call.

        A generated prose summary was the original design; it cost an extra
        request per PR to restate what the inline comments already say. Counts
        are more useful and free.
        """
        if not comments:
            return (
                "Reviewed "
                f"{result.files_reviewed} changed file(s) — nothing worth flagging."
            )

        counts = Counter(c.severity.value for c in comments)
        order = [Severity.CRITICAL, Severity.MAJOR, Severity.MINOR, Severity.NIT]
        breakdown = ", ".join(
            f"**{counts[s.value]} {s.value}**" for s in order if counts[s.value]
        )

        lines = [
            f"Reviewed {result.files_reviewed} changed file(s) and left "
            f"{len(comments)} comment(s): {breakdown}."
        ]

        if stats.get("over_cap"):
            lines.append(
                f"\n_{stats['over_cap']} further finding(s) were suppressed by the "
                "comment cap — the most severe are shown._"
            )
        if counts.get("critical"):
            lines.append("\n⚠️ Critical findings should be addressed before merge.")

        lines.append(
            "\n<sub>Automated review. Reply to a comment if it's wrong — "
            "false positives are worth knowing about.</sub>"
        )
        return "\n".join(lines)
