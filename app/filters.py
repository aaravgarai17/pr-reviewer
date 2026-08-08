"""Noise control: turning raw model output into comments worth reading.

Why this module is not optional
-------------------------------
An unfiltered LLM will happily produce forty comments on a two-hundred-line
pull request. Every one of them is individually plausible. Collectively they
are unusable — the author skims, finds the third one is a nitpick, and stops
reading. After two such reviews the bot gets muted, and a muted bot has
negative value: it consumed API budget and taught the team to ignore it.

So the pipeline is deliberately lossy, in this order:

  1. **Anchor** — drop anything that can't be placed on a changed line.
  2. **Threshold** — drop anything below the configured severity.
  3. **Deduplicate** — collapse the same point made repeatedly.
  4. **Cap** — keep the most severe N.

Order matters. Deduplicating before thresholding would let a `nit` survive as
the representative of a group whose `major` sibling was dropped, and capping
before sorting would keep whichever comments happened to arrive first.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from app.diff import FileDiff, LineKind
from app.models import Finding, PlacedComment, Severity

log = logging.getLogger("prbot.filters")


def anchor_findings(
    findings: list[Finding], files_by_path: dict[str, FileDiff]
) -> tuple[list[PlacedComment], list[tuple[Finding, str]]]:
    """Attach each finding to a real, commentable position in the diff.

    Returns `(placed, [(finding, reason)])`. Models occasionally cite a file
    that isn't in the PR or a line outside the diff; GitHub rejects the whole
    review payload if any comment is unplaceable, so these must be caught here
    rather than discovered at post time.
    """
    placed: list[PlacedComment] = []
    dropped: list[tuple[Finding, str]] = []

    for finding in findings:
        file = files_by_path.get(finding.file)

        if file is None:
            # Try a suffix match — models sometimes shorten paths.
            candidates = [
                f for p, f in files_by_path.items()
                if p.endswith("/" + finding.file) or finding.file.endswith("/" + p)
            ]
            if len(candidates) == 1:
                file = candidates[0]
            else:
                dropped.append((finding, f"file '{finding.file}' is not in this PR"))
                continue

        line = file.nearest_commentable_line(finding.line)
        if line is None:
            dropped.append(
                (finding, f"line {finding.line} is not an added line in the diff")
            )
            continue

        if line.kind is not LineKind.ADDED:
            dropped.append((finding, "target line was not added by this PR"))
            continue

        placed.append(
            PlacedComment(
                path=file.path,
                line=line.new_line,
                position=line.position,
                side="RIGHT",
                body=_format_body(finding),
                severity=finding.severity,
            )
        )

    return placed, dropped


def _format_body(finding: Finding) -> str:
    """Render a comment with a severity marker the eye can scan."""
    marker = {
        Severity.CRITICAL: "🔴 **critical**",
        Severity.MAJOR: "🟠 **major**",
        Severity.MINOR: "🟡 **minor**",
        Severity.NIT: "⚪ **nit**",
    }[finding.severity]

    return f"{marker} · _{finding.category}_\n\n{finding.comment}"


def apply_severity_threshold(
    comments: list[PlacedComment], threshold: Severity
) -> list[PlacedComment]:
    return [c for c in comments if c.severity.at_least(threshold)]


def _normalise(text: str) -> str:
    """Strip formatting so near-identical comments compare equal."""
    text = re.sub(r"[*_`#·]", " ", text.lower())
    text = re.sub(r"\b(critical|major|minor|nit)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def deduplicate(
    comments: list[PlacedComment], similarity_threshold: float = 0.85
) -> list[PlacedComment]:
    """Collapse comments that make the same point.

    Two sources of duplication in practice: the same issue repeated across
    chunks that both touched a shared helper, and the model restating one
    problem on several adjacent lines. Both read as the bot labouring a point.

    Comparison is quadratic in the number of comments, which is fine — the cap
    keeps that number small, and the alternative (embedding every comment)
    costs an API call to save microseconds.
    """
    kept: list[PlacedComment] = []

    for candidate in comments:
        norm = _normalise(candidate.body)
        duplicate = False

        for existing in kept:
            # Same line, same file is a duplicate regardless of wording.
            if existing.path == candidate.path and existing.line == candidate.line:
                duplicate = True
                break

            ratio = SequenceMatcher(None, norm, _normalise(existing.body)).ratio()
            if ratio >= similarity_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


def cap_comments(comments: list[PlacedComment], limit: int) -> list[PlacedComment]:
    """Keep the most severe `limit` comments, in file order.

    Sorting by severity to choose, then restoring file order to present, means
    the author reads top-to-bottom through their diff rather than jumping
    around — while still getting the most important findings when the cap bites.
    """
    chosen = comments
    if len(comments) > limit:
        # Choose by severity so the cap keeps what matters most...
        chosen = sorted(comments, key=lambda c: -c.severity.rank)[:limit]
        log.info("capping %d comments to %d", len(comments), limit)

    # ...but always present in file order, whether the cap bit or not, so the
    # author reads straight down their diff instead of jumping around.
    return sorted(chosen, key=lambda c: (c.path, c.line))


def filter_pipeline(
    findings: list[Finding],
    files_by_path: dict[str, FileDiff],
    threshold: Severity,
    max_comments: int,
    similarity_threshold: float = 0.85,
) -> tuple[list[PlacedComment], dict[str, int]]:
    """Run the full noise-control pipeline, reporting what each stage removed."""
    stats = {"raw": len(findings)}

    placed, dropped = anchor_findings(findings, files_by_path)
    stats["unanchorable"] = len(dropped)
    for finding, reason in dropped:
        log.debug("dropped finding on %s:%s — %s", finding.file, finding.line, reason)

    after_threshold = apply_severity_threshold(placed, threshold)
    stats["below_threshold"] = len(placed) - len(after_threshold)

    after_dedupe = deduplicate(after_threshold, similarity_threshold)
    stats["duplicates"] = len(after_threshold) - len(after_dedupe)

    final = cap_comments(after_dedupe, max_comments)
    stats["over_cap"] = len(after_dedupe) - len(final)
    stats["kept"] = len(final)

    return final, stats
