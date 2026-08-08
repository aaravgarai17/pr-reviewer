"""Unified diff parser, and the line-number mapping GitHub's API demands.

Why this module is the hardest part of the project
--------------------------------------------------
An LLM reviewing code says things like "line 42 dereferences a possibly-null
pointer". To turn that into an inline comment, GitHub needs to be told *where*
in the pull request to attach it — and it does not accept "line 42 of the file".

There are two accepted addressing schemes, and they are not the same thing:

  ``position``
      How many lines down you are from the **first** ``@@`` header in that
      file's diff. The line immediately after the first ``@@`` is position 1.
      Crucially the count includes context lines, removed lines, added lines,
      *and subsequent ``@@`` headers themselves*. It is an offset into the
      rendered diff, not into any file.

  ``line`` + ``side``
      The absolute line number in the file, plus whether that means the old
      version (``LEFT``) or the new one (``RIGHT``). Newer, far easier to
      reason about, and what this bot prefers.

Both are computed here, because they answer different questions and both are
easy to get subtly wrong. The classic bug is assuming position and new-file
line number are interchangeable — they agree only in the trivial case of a
single hunk starting at line 1 with no deletions, which is exactly the case
you'd hand-test with.

Consider a hunk header of ``@@ -10,7 +10,8 @@``. The first line under it is
diff position 1 but file line 10. Comment at position 10 and GitHub silently
places your remark nine lines from where you meant, or rejects it outright for
pointing at a line not in the diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

# @@ -old_start,old_count +new_start,new_count @@ optional section heading
HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<heading>.*)$"
)


class LineKind(str, Enum):
    CONTEXT = "context"     # unchanged, shown for orientation
    ADDED = "added"         # present only in the new version
    REMOVED = "removed"     # present only in the old version


class FileStatus(str, Enum):
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENAMED = "renamed"
    BINARY = "binary"


@dataclass
class DiffLine:
    kind: LineKind
    content: str
    position: int                      # offset from the file's first @@ header
    old_line: Optional[int] = None     # line number in the old file
    new_line: Optional[int] = None     # line number in the new file

    @property
    def is_commentable(self) -> bool:
        """Whether GitHub will accept an inline comment here.

        Only lines that are part of the diff can be commented on, which
        excludes nothing here — but added lines are the ones worth reviewing,
        since context lines weren't touched by this PR.
        """
        return self.kind in (LineKind.ADDED, LineKind.REMOVED)


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    heading: str
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def header(self) -> str:
        return (
            f"@@ -{self.old_start},{self.old_count} "
            f"+{self.new_start},{self.new_count} @@{self.heading}"
        )

    @property
    def added_lines(self) -> list[DiffLine]:
        return [l for l in self.lines if l.kind is LineKind.ADDED]

    def render(self) -> str:
        """Reconstruct this hunk as diff text, for sending to the model."""
        prefix = {LineKind.CONTEXT: " ", LineKind.ADDED: "+", LineKind.REMOVED: "-"}
        body = "\n".join(prefix[l.kind] + l.content for l in self.lines)
        return f"{self.header}\n{body}"


@dataclass
class FileDiff:
    path: str                          # the new path (or old, if deleted)
    old_path: Optional[str] = None     # differs from path only on rename
    status: FileStatus = FileStatus.MODIFIED
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def is_binary(self) -> bool:
        return self.status is FileStatus.BINARY

    @property
    def added_line_count(self) -> int:
        return sum(len(h.added_lines) for h in self.hunks)

    @property
    def total_line_count(self) -> int:
        return sum(len(h.lines) for h in self.hunks)

    def render(self) -> str:
        return "\n".join(h.render() for h in self.hunks)

    def line_at(self, new_line: int) -> Optional[DiffLine]:
        """Find the diff line corresponding to a line number in the new file.

        This is the lookup that turns a model's "line 42" into something
        GitHub will accept. Returns None when the model cites a line that
        isn't part of the diff — which happens, and must be handled rather
        than crashed on.
        """
        for hunk in self.hunks:
            for line in hunk.lines:
                if line.new_line == new_line:
                    return line
        return None

    def nearest_commentable_line(self, new_line: int) -> Optional[DiffLine]:
        """Best commentable line for a cited line number.

        Models sometimes cite a line just outside the diff — usually adjacent
        context they were shown. Rather than dropping the comment, snap to the
        closest added line in the same file, but only if it's close enough that
        the comment still plausibly refers to it.
        """
        exact = self.line_at(new_line)
        if exact is not None and exact.kind is LineKind.ADDED:
            return exact

        candidates = [
            l for h in self.hunks for l in h.lines if l.kind is LineKind.ADDED
        ]
        if not candidates:
            return None

        nearest = min(candidates, key=lambda l: abs((l.new_line or 0) - new_line))
        if abs((nearest.new_line or 0) - new_line) <= 3:
            return nearest
        return None


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff into structured files, hunks, and lines.

    Handles the cases that show up in real pull requests and break naive
    parsers: new files, deletions, renames, binary blobs, files with multiple
    hunks, and the ``\\ No newline at end of file`` marker.
    """
    if not diff_text or not diff_text.strip():
        return []

    files: list[FileDiff] = []
    current: Optional[FileDiff] = None
    hunk: Optional[Hunk] = None
    position = 0
    old_line = new_line = 0

    for raw in diff_text.splitlines():
        # --- new file header -------------------------------------------------
        if raw.startswith("diff --git "):
            if current is not None:
                files.append(current)
            current = _start_file(raw)
            hunk = None
            position = 0
            continue

        if current is None:
            continue          # preamble before the first file; ignore

        # --- file-level metadata --------------------------------------------
        if raw.startswith("rename from "):
            current.old_path = raw[len("rename from "):].strip()
            current.status = FileStatus.RENAMED
            continue
        if raw.startswith("rename to "):
            current.path = raw[len("rename to "):].strip()
            current.status = FileStatus.RENAMED
            continue
        if raw.startswith("new file mode"):
            current.status = FileStatus.ADDED
            continue
        if raw.startswith("deleted file mode"):
            current.status = FileStatus.DELETED
            continue
        if raw.startswith("Binary files ") or raw.startswith("GIT binary patch"):
            current.status = FileStatus.BINARY
            continue
        if raw.startswith("index ") or raw.startswith("similarity index"):
            continue

        if raw.startswith("--- "):
            path = raw[4:].strip()
            if path == "/dev/null":
                current.status = FileStatus.ADDED
            else:
                current.old_path = current.old_path or _strip_prefix(path)
            continue

        if raw.startswith("+++ "):
            path = raw[4:].strip()
            if path == "/dev/null":
                current.status = FileStatus.DELETED
            else:
                current.path = _strip_prefix(path)
            continue

        # --- hunk header -----------------------------------------------------
        match = HUNK_HEADER.match(raw)
        if match:
            hunk = Hunk(
                old_start=int(match.group("old_start")),
                old_count=int(match.group("old_count") or 1),
                new_start=int(match.group("new_start")),
                new_count=int(match.group("new_count") or 1),
                heading=match.group("heading") or "",
            )
            current.hunks.append(hunk)

            old_line = hunk.old_start
            new_line = hunk.new_start

            # A second or later @@ header still occupies a slot in GitHub's
            # position counting. Missing this shifts every following comment.
            if len(current.hunks) > 1:
                position += 1
            continue

        if hunk is None:
            continue          # content outside any hunk

        # --- hunk body -------------------------------------------------------
        if raw.startswith("\\"):
            # "\ No newline at end of file" — metadata, not a diff line, and
            # explicitly does not advance the position counter.
            continue

        position += 1
        marker, content = (raw[:1], raw[1:]) if raw else (" ", "")

        if marker == "+":
            hunk.lines.append(
                DiffLine(LineKind.ADDED, content, position, None, new_line)
            )
            new_line += 1
        elif marker == "-":
            hunk.lines.append(
                DiffLine(LineKind.REMOVED, content, position, old_line, None)
            )
            old_line += 1
        else:
            hunk.lines.append(
                DiffLine(LineKind.CONTEXT, content, position, old_line, new_line)
            )
            old_line += 1
            new_line += 1

    if current is not None:
        files.append(current)

    return files


def _start_file(header_line: str) -> FileDiff:
    """Derive the path from a ``diff --git a/x b/x`` line.

    The +++ header that follows is authoritative and will overwrite this; the
    fallback matters for binary files, which have no +++ line at all.
    """
    parts = header_line.split(" b/", 1)
    path = parts[1].strip() if len(parts) == 2 else header_line
    return FileDiff(path=path)


def _strip_prefix(path: str) -> str:
    """Remove git's a/ or b/ prefix."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def iter_reviewable_files(files: list[FileDiff]) -> Iterator[FileDiff]:
    """Files worth sending to a model.

    Binary blobs have nothing to read, and deleted files have no code left to
    critique — commenting on either wastes tokens and produces noise.
    """
    for f in files:
        if f.is_binary or f.status is FileStatus.DELETED:
            continue
        if not f.hunks:
            continue
        yield f
