"""Splitting a pull request into requests that fit the context window.

The constraint
--------------
A 5,000-line pull request will not fit in one call, and even where it would
fit, it shouldn't: attention degrades over very long inputs, and a single
failed call loses the whole review.

The strategy, in order of preference:

  1. **Pack whole files together** while they fit the budget. Reviewing a file
     in one piece gives the model the most coherent view of it.
  2. **A file alone** if it fills a request by itself.
  3. **Split by hunk** if even one file is too large. Hunks are natural
     boundaries — git already chose them as coherent regions of change.
  4. **Skip** a single hunk that exceeds the budget on its own. Truncating
     mid-function produces confident nonsense about code the model can't see,
     which is worse than silence.

Why not send whole files with surrounding context
-------------------------------------------------
More context genuinely improves review quality, and it costs linearly. A
50-file PR at 2,000 tokens of context each is 100K tokens of input for a review
that might only need the 400 changed lines. The diff plus the few context lines
git already includes is the point on that curve where marginal quality stops
justifying marginal cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.diff import FileDiff, Hunk
from app.llm import estimate_tokens
from app.prompts import render_file_for_review

log = logging.getLogger("prbot.chunking")


@dataclass
class Chunk:
    """One LLM request's worth of diff."""

    rendered: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    estimated_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.rendered


@dataclass
class ChunkPlan:
    chunks: list[Chunk] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (path, reason)

    @property
    def total_tokens(self) -> int:
        return sum(c.estimated_tokens for c in self.chunks)


def plan_chunks(
    files: list[FileDiff],
    budget_tokens: int = 12_000,
    max_files_per_chunk: int = 10,
) -> ChunkPlan:
    """Group files into requests that each fit within `budget_tokens`."""
    plan = ChunkPlan()
    current = Chunk()

    def flush() -> None:
        nonlocal current
        if not current.is_empty:
            plan.chunks.append(current)
            current = Chunk()

    for file in files:
        rendered = render_file_for_review(file)
        cost = estimate_tokens(rendered)

        if cost > budget_tokens:
            # Too big whole; fall back to hunk-level splitting.
            flush()
            _split_file_by_hunks(file, budget_tokens, plan)
            continue

        would_exceed = current.estimated_tokens + cost > budget_tokens
        too_many = len(current.files) >= max_files_per_chunk

        if not current.is_empty and (would_exceed or too_many):
            flush()

        current.rendered.append(rendered)
        current.files.append(file.path)
        current.estimated_tokens += cost

    flush()
    return plan


def _split_file_by_hunks(file: FileDiff, budget: int, plan: ChunkPlan) -> None:
    """Emit one or more chunks for a file too large to send whole."""
    current = Chunk()
    batch: list[Hunk] = []

    def flush() -> None:
        nonlocal current, batch
        if batch:
            rendered = render_file_for_review(file, batch)
            current.rendered.append(rendered)
            current.files.append(file.path)
            current.estimated_tokens += estimate_tokens(rendered)
            plan.chunks.append(current)
        current = Chunk()
        batch = []

    for hunk in file.hunks:
        cost = estimate_tokens(hunk.render())

        if cost > budget:
            # One hunk larger than a whole request. Truncating it would show
            # the model half a function and invite confident nonsense about
            # the half it cannot see.
            flush()
            plan.skipped.append(
                (file.path, f"a single hunk exceeds the {budget}-token budget")
            )
            log.warning("skipping oversized hunk in %s", file.path)
            continue

        if batch and current.estimated_tokens + cost > budget:
            flush()

        batch.append(hunk)
        current.estimated_tokens += cost

    flush()


def filter_reviewable(
    files: list[FileDiff],
    ignore_patterns: list[str],
    max_file_lines: int,
) -> tuple[list[FileDiff], list[tuple[str, str]]]:
    """Drop files not worth reviewing. Returns (kept, [(path, reason)]).

    Generated files, lockfiles, and vendored dependencies are the main target.
    Nobody hand-wrote `package-lock.json`, nobody will act on a comment about
    it, and it can be tens of thousands of lines — so reviewing it is pure cost
    for zero value.
    """
    import fnmatch

    kept: list[FileDiff] = []
    skipped: list[tuple[str, str]] = []

    for file in files:
        if file.is_binary:
            skipped.append((file.path, "binary"))
            continue

        if file.status.value == "deleted":
            skipped.append((file.path, "file deleted"))
            continue

        if not file.hunks:
            skipped.append((file.path, "no reviewable changes"))
            continue

        matched = next(
            (p for p in ignore_patterns if fnmatch.fnmatch(file.path, p)), None
        )
        if matched:
            skipped.append((file.path, f"matches ignore pattern '{matched}'"))
            continue

        if file.total_line_count > max_file_lines:
            skipped.append(
                (file.path, f"{file.total_line_count} lines exceeds limit of {max_file_lines}")
            )
            continue

        kept.append(file)

    return kept, skipped
