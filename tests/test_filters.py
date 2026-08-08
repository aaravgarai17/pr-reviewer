"""Noise control: anchoring, thresholds, deduplication, capping."""

import pytest

from app.diff import parse_diff
from app.filters import (
    anchor_findings,
    apply_severity_threshold,
    cap_comments,
    deduplicate,
    filter_pipeline,
)
from app.models import Finding, PlacedComment, Severity

DIFF = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,4 +1,7 @@
 import os
+import sys
+
+def risky(user):
+    return user.name
 x = 1
"""


@pytest.fixture
def files_by_path():
    return {f.path: f for f in parse_diff(DIFF)}


def finding(line=2, severity="major", file="app.py", comment="Something is wrong."):
    return Finding(file=file, line=line, severity=severity, comment=comment)


def placed(line=2, severity=Severity.MAJOR, path="app.py", body="text"):
    return PlacedComment(
        path=path, line=line, position=line, side="RIGHT", body=body, severity=severity
    )


# ------------------------------------------------------------------ anchoring


def test_anchors_to_an_added_line(files_by_path):
    placed_comments, dropped = anchor_findings([finding(line=2)], files_by_path)

    assert len(placed_comments) == 1
    assert dropped == []
    assert placed_comments[0].path == "app.py"
    assert placed_comments[0].line == 2


def test_drops_findings_for_files_not_in_the_pr(files_by_path):
    _, dropped = anchor_findings([finding(file="nonexistent.py")], files_by_path)

    assert len(dropped) == 1
    assert "not in this PR" in dropped[0][1]


def test_matches_a_shortened_path(files_by_path):
    """Models sometimes cite a bare filename instead of the full path."""
    files = {"src/deep/app.py": list(files_by_path.values())[0]}
    files["src/deep/app.py"].path = "src/deep/app.py"

    placed_comments, dropped = anchor_findings([finding(file="app.py")], files)
    assert len(placed_comments) == 1


def test_drops_findings_far_outside_the_diff(files_by_path):
    _, dropped = anchor_findings([finding(line=9999)], files_by_path)

    assert len(dropped) == 1
    assert "not an added line" in dropped[0][1]


def test_snaps_a_near_miss_to_a_real_added_line(files_by_path):
    """A citation one or two lines off is worth rescuing, not discarding."""
    placed_comments, _ = anchor_findings([finding(line=4)], files_by_path)
    assert len(placed_comments) == 1


def test_body_includes_a_severity_marker(files_by_path):
    placed_comments, _ = anchor_findings(
        [finding(severity="critical", comment="SQL injection here.")], files_by_path
    )
    body = placed_comments[0].body
    assert "critical" in body
    assert "SQL injection here." in body


def test_unanchorable_findings_do_not_break_the_batch(files_by_path):
    """GitHub rejects an entire review if any one comment is unplaceable, so
    bad ones must be filtered here rather than discovered at post time."""
    findings = [finding(line=2), finding(line=9999), finding(line=3)]
    placed_comments, dropped = anchor_findings(findings, files_by_path)

    assert len(placed_comments) == 2
    assert len(dropped) == 1


# ------------------------------------------------------------------ threshold


def test_threshold_keeps_equal_and_above():
    comments = [
        placed(severity=Severity.NIT),
        placed(severity=Severity.MINOR),
        placed(severity=Severity.MAJOR),
        placed(severity=Severity.CRITICAL),
    ]
    kept = apply_severity_threshold(comments, Severity.MINOR)
    assert [c.severity for c in kept] == [
        Severity.MINOR, Severity.MAJOR, Severity.CRITICAL
    ]


def test_threshold_critical_only():
    comments = [placed(severity=Severity.MAJOR), placed(severity=Severity.CRITICAL)]
    kept = apply_severity_threshold(comments, Severity.CRITICAL)
    assert len(kept) == 1


def test_severity_ordering():
    assert Severity.CRITICAL.at_least(Severity.NIT)
    assert not Severity.NIT.at_least(Severity.CRITICAL)
    assert Severity.MAJOR.at_least(Severity.MAJOR)


# --------------------------------------------------------------- deduplication


def test_same_file_and_line_is_a_duplicate():
    comments = [
        placed(line=5, body="This could be None."),
        placed(line=5, body="Completely different wording about something else."),
    ]
    assert len(deduplicate(comments)) == 1


def test_near_identical_wording_is_a_duplicate():
    comments = [
        placed(line=5, body="This variable may be None and will raise."),
        placed(line=9, body="This variable may be None and will raise!"),
    ]
    assert len(deduplicate(comments)) == 1


def test_genuinely_different_comments_are_kept():
    comments = [
        placed(line=5, body="Possible null dereference here."),
        placed(line=9, body="This loop performs a database query per iteration."),
    ]
    assert len(deduplicate(comments)) == 2


def test_dedupe_ignores_severity_markers_when_comparing():
    comments = [
        placed(line=5, severity=Severity.MAJOR, body="🟠 **major** · _bug_\n\nNull check missing."),
        placed(line=9, severity=Severity.MINOR, body="🟡 **minor** · _bug_\n\nNull check missing."),
    ]
    assert len(deduplicate(comments)) == 1


# ---------------------------------------------------------------- capping


def test_cap_keeps_everything_below_the_limit():
    comments = [placed(line=i) for i in range(1, 4)]
    assert len(cap_comments(comments, 10)) == 3


def test_file_order_applies_even_when_the_cap_does_not_bite():
    """Ordering must be consistent whether or not comments were dropped."""
    comments = [
        placed(path="z.py", line=1),
        placed(path="a.py", line=5),
        placed(path="a.py", line=2),
    ]
    kept = cap_comments(comments, 100)
    assert [(c.path, c.line) for c in kept] == [("a.py", 2), ("a.py", 5), ("z.py", 1)]


def test_cap_keeps_the_most_severe():
    comments = [
        placed(line=1, severity=Severity.NIT),
        placed(line=2, severity=Severity.CRITICAL),
        placed(line=3, severity=Severity.MINOR),
        placed(line=4, severity=Severity.MAJOR),
    ]
    kept = cap_comments(comments, 2)

    assert len(kept) == 2
    assert {c.severity for c in kept} == {Severity.CRITICAL, Severity.MAJOR}


def test_capped_comments_are_returned_in_file_order():
    """Chosen by severity, presented in reading order."""
    comments = [
        placed(path="z.py", line=1, severity=Severity.CRITICAL),
        placed(path="a.py", line=5, severity=Severity.CRITICAL),
        placed(path="a.py", line=2, severity=Severity.CRITICAL),
    ]
    kept = cap_comments(comments, 3)
    assert [(c.path, c.line) for c in kept] == [("a.py", 2), ("a.py", 5), ("z.py", 1)]


# ---------------------------------------------------------------- pipeline


def test_full_pipeline_reports_each_stage(files_by_path):
    findings = [
        finding(line=2, severity="critical", comment="Real problem here."),
        finding(line=3, severity="nit", comment="Tiny style thing."),
        finding(line=9999, severity="major", comment="Points outside the diff."),
        finding(line=4, severity="major", comment="Real problem here."),
    ]

    comments, stats = filter_pipeline(
        findings, files_by_path, threshold=Severity.MINOR, max_comments=10
    )

    assert stats["raw"] == 4
    assert stats["unanchorable"] == 1        # line 9999
    assert stats["below_threshold"] == 1     # the nit
    assert stats["duplicates"] == 1          # same wording as the first
    assert stats["kept"] == 1


def test_pipeline_with_nothing_to_report(files_by_path):
    comments, stats = filter_pipeline([], files_by_path, Severity.MINOR, 10)
    assert comments == []
    assert stats["kept"] == 0


def test_pipeline_respects_the_cap(files_by_path):
    findings = [
        finding(line=2, severity="major", comment="First distinct issue."),
        finding(line=3, severity="major", comment="Second unrelated concern entirely."),
        finding(line=4, severity="major", comment="Third totally separate matter here."),
    ]
    comments, stats = filter_pipeline(
        findings, files_by_path, threshold=Severity.MINOR, max_comments=2
    )

    assert len(comments) == 2
    assert stats["over_cap"] == 1
