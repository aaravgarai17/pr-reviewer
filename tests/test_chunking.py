"""Context-budget chunking and file filtering."""

import pytest

from app.chunking import filter_reviewable, plan_chunks
from app.diff import parse_diff


def make_diff(path: str, added_lines: int, hunks: int = 1) -> str:
    """Build a synthetic diff of roughly known size."""
    out = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}"]
    line_no = 1
    for h in range(hunks):
        start = 1 + h * 1000
        out.append(f"@@ -{start},1 +{start},{added_lines + 1} @@")
        out.append(" context line")
        for i in range(added_lines):
            out.append(f"+    some_function_call(argument_{i}, another_argument_{i})")
    return "\n".join(out) + "\n"


def test_small_files_are_packed_together():
    diff = make_diff("a.py", 5) + make_diff("b.py", 5) + make_diff("c.py", 5)
    plan = plan_chunks(parse_diff(diff), budget_tokens=100_000)

    assert len(plan.chunks) == 1
    assert plan.chunks[0].files == ["a.py", "b.py", "c.py"]


def test_files_split_across_chunks_when_the_budget_is_small():
    """Each file fits alone (~885 tokens) but no two fit together."""
    diff = make_diff("a.py", 40) + make_diff("b.py", 40) + make_diff("c.py", 40)
    plan = plan_chunks(parse_diff(diff), budget_tokens=1200)

    assert len(plan.chunks) == 3
    assert plan.skipped == []
    packed = [f for c in plan.chunks for f in c.files]
    assert set(packed) == {"a.py", "b.py", "c.py"}


def test_no_chunk_exceeds_the_budget():
    diff = "".join(make_diff(f"f{i}.py", 20) for i in range(10))
    budget = 1500
    plan = plan_chunks(parse_diff(diff), budget_tokens=budget)

    for chunk in plan.chunks:
        assert chunk.estimated_tokens <= budget


def test_max_files_per_chunk_is_respected():
    diff = "".join(make_diff(f"f{i}.py", 2) for i in range(12))
    plan = plan_chunks(parse_diff(diff), budget_tokens=100_000, max_files_per_chunk=5)

    assert all(len(c.files) <= 5 for c in plan.chunks)
    assert len(plan.chunks) == 3


def test_a_large_file_is_split_by_hunk():
    """A file too big to send whole falls back to hunk-level splitting."""
    diff = make_diff("big.py", 30, hunks=6)
    plan = plan_chunks(parse_diff(diff), budget_tokens=1200)

    assert len(plan.chunks) > 1
    assert all("big.py" in c.files for c in plan.chunks)


def test_a_single_oversized_hunk_is_skipped_not_truncated():
    """Truncating mid-function invites confident nonsense about unseen code."""
    diff = make_diff("huge.py", 500, hunks=1)
    plan = plan_chunks(parse_diff(diff), budget_tokens=200)

    assert plan.skipped
    assert plan.skipped[0][0] == "huge.py"
    assert "exceeds" in plan.skipped[0][1]


def test_empty_input():
    plan = plan_chunks([], budget_tokens=1000)
    assert plan.chunks == []
    assert plan.total_tokens == 0


def test_every_file_appears_somewhere():
    """Nothing may be silently lost during chunking."""
    diff = "".join(make_diff(f"f{i}.py", 15) for i in range(8))
    files = parse_diff(diff)
    plan = plan_chunks(files, budget_tokens=900)

    placed = {f for c in plan.chunks for f in c.files}
    skipped = {p for p, _ in plan.skipped}
    assert placed | skipped == {f.path for f in files}


# ------------------------------------------------------------------ filtering


def test_ignore_patterns_drop_generated_files():
    diff = (
        make_diff("src/app.py", 3)
        + make_diff("package-lock.json", 3)
        + make_diff("dist/bundle.min.js", 3)
    )
    kept, skipped = filter_reviewable(
        parse_diff(diff),
        ignore_patterns=["package-lock.json", "*.min.js", "dist/*"],
        max_file_lines=1000,
    )

    assert [f.path for f in kept] == ["src/app.py"]
    assert len(skipped) == 2


def test_binary_files_are_skipped():
    diff = """diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
"""
    kept, skipped = filter_reviewable(parse_diff(diff), [], 1000)
    assert kept == []
    assert skipped[0][1] == "binary"


def test_deleted_files_are_skipped():
    """No code left to critique."""
    diff = """diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-x = 1
-y = 2
"""
    kept, skipped = filter_reviewable(parse_diff(diff), [], 1000)
    assert kept == []
    assert "deleted" in skipped[0][1]


def test_oversized_files_are_skipped():
    diff = make_diff("massive.py", 400)
    kept, skipped = filter_reviewable(parse_diff(diff), [], max_file_lines=100)

    assert kept == []
    assert "exceeds limit" in skipped[0][1]


def test_skip_reasons_are_recorded():
    """Reasons are logged and surfaced, not swallowed."""
    diff = make_diff("yarn.lock", 5)
    _, skipped = filter_reviewable(parse_diff(diff), ["yarn.lock"], 1000)

    assert skipped[0][0] == "yarn.lock"
    assert "ignore pattern" in skipped[0][1]
