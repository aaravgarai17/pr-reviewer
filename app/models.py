"""Shared data shapes."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    """Ordered by how much a reviewer should care.

    Definitions live in the prompt too — an LLM given only the labels will
    invent its own thresholds and mark everything `major`, which defeats the
    filtering these levels exist to enable.
    """

    CRITICAL = "critical"   # data loss, security hole, guaranteed crash
    MAJOR = "major"         # real bug under plausible conditions
    MINOR = "minor"         # smell, edge case, unclear naming
    NIT = "nit"             # cosmetic preference

    @property
    def rank(self) -> int:
        return {"critical": 3, "major": 2, "minor": 1, "nit": 0}[self.value]

    def at_least(self, threshold: "Severity") -> bool:
        return self.rank >= threshold.rank


class Finding(BaseModel):
    """One review remark, before it has been placed in the diff."""

    file: str
    line: int = Field(..., description="Line number in the *new* version")
    severity: Severity
    comment: str
    category: str = "general"

    @field_validator("comment")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("comment must not be empty")
        return v.strip()

    @field_validator("line")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("line must be positive")
        return v


class PlacedComment(BaseModel):
    """A finding successfully anchored to a real position in the diff."""

    path: str
    line: int              # new-file line number, for the line/side API
    position: int          # offset from the first @@, for the legacy API
    side: str = "RIGHT"
    body: str
    severity: Severity

    def to_github(self) -> dict:
        """The payload GitHub's Review API expects.

        Uses `line` + `side` rather than `position`: it is the newer form and
        far less error-prone. `position` is carried alongside for reference and
        for the legacy endpoint.
        """
        return {"path": self.path, "line": self.line, "side": self.side, "body": self.body}


class PullRequestRef(BaseModel):
    """Everything needed to fetch and comment on a PR."""

    owner: str
    repo: str
    number: int
    head_sha: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    def __str__(self) -> str:
        return f"{self.full_name}#{self.number}@{self.head_sha[:7]}"


class ReviewResult(BaseModel):
    """Outcome of a review run, for logging and tests."""

    pr: PullRequestRef
    findings_raw: int = 0          # what the model produced
    findings_kept: int = 0         # what survived filtering
    comments_posted: int = 0       # what was anchored and sent
    files_reviewed: int = 0
    files_skipped: int = 0
    chunks_sent: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    skipped_reason: Optional[str] = None

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost at Claude Sonnet pricing ($3/MTok in, $15/MTok out)."""
        return (self.input_tokens / 1e6) * 3.0 + (self.output_tokens / 1e6) * 15.0

    def summary_line(self) -> str:
        return (
            f"{self.pr}: {self.comments_posted} comment(s) from "
            f"{self.findings_raw} finding(s) across {self.files_reviewed} file(s), "
            f"{self.chunks_sent} LLM call(s), ~${self.estimated_cost_usd:.4f}"
        )
