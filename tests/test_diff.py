"""Diff parsing and position mapping.

The position tests are the important ones. Comments land on the wrong lines
when position accounting drifts, and the failure is silent — GitHub happily
attaches your remark somewhere else.
"""

import pytest

from app.diff import (
    FileStatus,
    LineKind,
    iter_reviewable_files,
    parse_diff,
)

SIMPLE = """diff --git a/hello.py b/hello.py
index 1234567..89abcde 100644
--- a/hello.py
+++ b/hello.py
@@ -1,4 +1,5 @@
 def greet(name):
-    print("hi")
+    print(f"hello {name}")
+    return None

 x = 1
"""

# Two hunks, and the second does not start at line 1 — the shape that exposes
# position/line-number confusion.
TWO_HUNKS = """diff --git a/app.py b/app.py
index 111..222 100644
--- a/app.py
+++ b/app.py
@@ -10,6 +10,7 @@ def first():
     a = 1
     b = 2
-    return a
+    c = 3
+    return a + c

 def spacer():
@@ -50,4 +51,5 @@ def second():
     x = 10
-    return x
+    y = 20
+    return x + y
"""


def test_parses_a_single_file():
    files = parse_diff(SIMPLE)
    assert len(files) == 1
    assert files[0].path == "hello.py"
    assert files[0].status is FileStatus.MODIFIED


def test_empty_diff():
    assert parse_diff("") == []
    assert parse_diff("   \n  ") == []


def test_line_kinds():
    hunk = parse_diff(SIMPLE)[0].hunks[0]
    kinds = [l.kind for l in hunk.lines]
    assert kinds == [
        LineKind.CONTEXT,
        LineKind.REMOVED,
        LineKind.ADDED,
        LineKind.ADDED,
        LineKind.CONTEXT,
        LineKind.CONTEXT,
    ]


def test_line_content_excludes_the_marker():
    hunk = parse_diff(SIMPLE)[0].hunks[0]
    added = hunk.added_lines
    assert added[0].content == '    print(f"hello {name}")'
    assert added[1].content == "    return None"


def test_hunk_header_is_parsed():
    hunk = parse_diff(SIMPLE)[0].hunks[0]
    assert (hunk.old_start, hunk.old_count) == (1, 4)
    assert (hunk.new_start, hunk.new_count) == (1, 5)


def test_hunk_heading_captured():
    """The text after @@ is the enclosing function — useful context."""
    hunk = parse_diff(TWO_HUNKS)[0].hunks[0]
    assert "def first" in hunk.heading


# ----------------------------------------------------- position accounting


def test_position_starts_at_one_under_the_hunk_header():
    lines = parse_diff(SIMPLE)[0].hunks[0].lines
    assert lines[0].position == 1
    assert lines[1].position == 2


def test_position_counts_every_line_type():
    """Context, added, and removed lines all advance the position counter."""
    lines = parse_diff(SIMPLE)[0].hunks[0].lines
    assert [l.position for l in lines] == [1, 2, 3, 4, 5, 6]


def test_position_and_file_line_number_are_different_things():
    """The trap this module exists to avoid.

    In the second hunk (starting at file line 51), position and new_line
    diverge sharply. Treating them as interchangeable puts comments dozens of
    lines from where they belong.
    """
    f = parse_diff(TWO_HUNKS)[0]
    second = f.hunks[1]
    first_line = second.lines[0]

    assert first_line.new_line == 51        # where it is in the file
    assert first_line.position == 9         # where it is in the diff
    assert first_line.position != first_line.new_line

    # A comment intended for file line 51, posted at position 51, would be
    # rejected outright — the diff has only 12 positions.
    assert f.hunks[-1].lines[-1].position == 12


def test_second_hunk_header_consumes_a_position():
    """GitHub counts subsequent @@ headers as diff lines.

    Skip this and every comment after the first hunk is off by one per hunk.
    """
    f = parse_diff(TWO_HUNKS)[0]
    last_of_first = f.hunks[0].lines[-1]
    first_of_second = f.hunks[1].lines[0]

    # +1 for the line itself, +1 for the intervening @@ header.
    assert first_of_second.position == last_of_first.position + 2


def test_line_numbers_track_independently_for_old_and_new():
    lines = parse_diff(SIMPLE)[0].hunks[0].lines

    context = lines[0]
    assert context.old_line == 1 and context.new_line == 1

    removed = lines[1]
    assert removed.old_line == 2 and removed.new_line is None

    added = lines[2]
    assert added.old_line is None and added.new_line == 2


def test_line_numbers_resume_correctly_after_a_deletion():
    """A removed line advances the old counter only, leaving new behind."""
    lines = parse_diff(SIMPLE)[0].hunks[0].lines
    trailing_context = lines[4]
    assert trailing_context.old_line == 3
    assert trailing_context.new_line == 4


def test_hunk_without_explicit_counts():
    """`@@ -1 +1 @@` means a count of 1, which must be defaulted."""
    diff = """diff --git a/x.txt b/x.txt
--- a/x.txt
+++ b/x.txt
@@ -1 +1 @@
-old
+new
"""
    hunk = parse_diff(diff)[0].hunks[0]
    assert hunk.old_count == 1 and hunk.new_count == 1


def test_no_newline_marker_does_not_advance_position():
    """`\\ No newline at end of file` is metadata, not a diff line."""
    diff = """diff --git a/x.txt b/x.txt
--- a/x.txt
+++ b/x.txt
@@ -1,2 +1,2 @@
 keep
-old
\\ No newline at end of file
+new
"""
    lines = parse_diff(diff)[0].hunks[0].lines
    assert [l.position for l in lines] == [1, 2, 3]
    assert lines[2].content == "new"


# --------------------------------------------------------------- file kinds


def test_new_file():
    diff = """diff --git a/new.py b/new.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+import os
+print(os.name)
"""
    f = parse_diff(diff)[0]
    assert f.status is FileStatus.ADDED
    assert f.path == "new.py"
    assert f.added_line_count == 2


def test_deleted_file():
    diff = """diff --git a/gone.py b/gone.py
deleted file mode 100644
index abc1234..0000000
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-import os
-print(os.name)
"""
    f = parse_diff(diff)[0]
    assert f.status is FileStatus.DELETED


def test_renamed_file():
    diff = """diff --git a/old_name.py b/new_name.py
similarity index 95%
rename from old_name.py
rename to new_name.py
index 111..222 100644
--- a/old_name.py
+++ b/new_name.py
@@ -1,3 +1,3 @@
 import os
-x = 1
+x = 2
 y = 3
"""
    f = parse_diff(diff)[0]
    assert f.status is FileStatus.RENAMED
    assert f.old_path == "old_name.py"
    assert f.path == "new_name.py"


def test_binary_file():
    diff = """diff --git a/logo.png b/logo.png
index 111..222 100644
Binary files a/logo.png and b/logo.png differ
"""
    f = parse_diff(diff)[0]
    assert f.is_binary
    assert f.hunks == []


def test_multiple_files():
    diff = SIMPLE + """diff --git a/other.py b/other.py
index 333..444 100644
--- a/other.py
+++ b/other.py
@@ -1,2 +1,2 @@
-a = 1
+a = 2
 b = 3
"""
    files = parse_diff(diff)
    assert [f.path for f in files] == ["hello.py", "other.py"]


def test_position_resets_between_files():
    """Position counts from the first @@ *of that file*, not the whole diff."""
    diff = SIMPLE + """diff --git a/other.py b/other.py
--- a/other.py
+++ b/other.py
@@ -1,2 +1,2 @@
-a = 1
+a = 2
"""
    files = parse_diff(diff)
    assert files[1].hunks[0].lines[0].position == 1


def test_paths_with_spaces():
    diff = """diff --git a/my folder/file name.py b/my folder/file name.py
--- a/my folder/file name.py
+++ b/my folder/file name.py
@@ -1 +1 @@
-a
+b
"""
    assert parse_diff(diff)[0].path == "my folder/file name.py"


# ------------------------------------------------------------ line lookup


def test_line_at_finds_by_new_file_line_number():
    f = parse_diff(TWO_HUNKS)[0]

    assert f.line_at(51).content == "    x = 10"      # context line
    assert f.line_at(52).content == "    y = 20"      # added line
    assert f.line_at(52).kind is LineKind.ADDED


def test_line_at_returns_none_outside_the_diff():
    f = parse_diff(SIMPLE)[0]
    assert f.line_at(9999) is None


def test_nearest_commentable_snaps_to_a_close_added_line():
    """Models sometimes cite an adjacent line; snap rather than drop."""
    f = parse_diff(SIMPLE)[0]
    line = f.nearest_commentable_line(4)
    assert line is not None
    assert line.kind is LineKind.ADDED


def test_nearest_commentable_gives_up_when_far_away():
    """Snapping a wildly wrong line number would place a nonsense comment."""
    f = parse_diff(SIMPLE)[0]
    assert f.nearest_commentable_line(500) is None


def test_line_at_exact_added_line_is_preferred():
    f = parse_diff(SIMPLE)[0]
    line = f.nearest_commentable_line(2)
    assert line.new_line == 2
    assert line.content == '    print(f"hello {name}")'


# ------------------------------------------------------------- rendering


def test_render_round_trips_a_hunk():
    hunk = parse_diff(SIMPLE)[0].hunks[0]
    rendered = hunk.render()

    assert rendered.startswith("@@ -1,4 +1,5 @@")
    assert '+    print(f"hello {name}")' in rendered
    assert '-    print("hi")' in rendered
    assert " def greet(name):" in rendered


# ------------------------------------------------------------- filtering


def test_reviewable_skips_binary_and_deleted():
    diff = SIMPLE + """diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
"""
    reviewable = list(iter_reviewable_files(parse_diff(diff)))
    assert [f.path for f in reviewable] == ["hello.py"]


def test_counts():
    f = parse_diff(SIMPLE)[0]
    assert f.added_line_count == 2
    assert f.total_line_count == 6
