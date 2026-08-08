"""The review prompt, and the reasoning behind each constraint in it.

Every rule below exists because its absence produced a specific bad behaviour.
The iteration story is documented in the README; this file records the
conclusions.

  1. **"Return [] if the code is fine."**  Without an explicit licence to say
     nothing, the model treats an empty response as failure and manufactures
     filler — "consider adding a comment here" on a two-line change. A bot that
     always finds something is a bot people mute.

  2. **"Do not comment on formatting."**  Left alone, roughly half of all
     output was quoting, spacing, and import order. Linters do that better,
     deterministically, and for free. Every style remark spends reviewer
     attention that a real bug needed.

  3. **Severity definitions, not just labels.**  Given bare labels the model
     rated almost everything `major`, which makes filtering useless. Concrete
     thresholds ("critical means data loss or a security hole") produce a
     usable distribution.

  4. **Cite the line number shown in the margin.**  The model is given the diff
     with explicit line numbers because asking it to count lines itself
     produces off-by-a-few errors, and a comment on the wrong line is worse
     than no comment.

  5. **JSON only, no prose, no code fences.**  Markdown fences around JSON were
     the single most common parse failure. Saying so explicitly, and stripping
     them defensively anyway, removed the problem.

  6. **"Only comment on added lines."**  Remarks on unchanged context are
     noise: the author didn't write that code in this PR and usually can't act
     on it.
"""

from __future__ import annotations

from app.diff import FileDiff, Hunk, LineKind

SYSTEM_PROMPT = """You are an experienced software engineer reviewing a pull \
request. You are thorough but not pedantic, and your time is valuable, so you \
only speak up when it matters.

WHAT TO LOOK FOR, in priority order:
1. Bugs — logic errors, off-by-one, null/None dereference, unhandled \
exceptions, incorrect conditionals, race conditions, resource leaks.
2. Security — injection, missing authorization checks, secrets in code, unsafe \
deserialization, path traversal.
3. Performance — N+1 queries, unbounded memory growth, needless work in a hot \
loop, blocking I/O on an async path.
4. Correctness risks — silent failure, swallowed exceptions, misleading names \
that will cause a future mistake.

WHAT TO IGNORE COMPLETELY:
- Formatting, whitespace, quote style, line length, import ordering. \
Automated linters handle these and do it better.
- Stylistic preferences that do not change behaviour.
- Requests to add comments or docstrings, unless the code is genuinely \
inscrutable without them.
- Anything on a line the author did not change in this pull request.

SEVERITY — apply these thresholds strictly:
- "critical": data loss, a security vulnerability, or a crash that will \
certainly happen in normal use.
- "major": a real bug that will occur under plausible conditions, or a \
significant performance problem.
- "minor": a genuine edge case, a resource that should be closed, a name that \
actively misleads.
- "nit": small and cosmetic but still substantive. Use sparingly.

Do not inflate severity. Most findings in real code are "minor". If you mark \
everything "major", the severity field becomes useless and reviewers stop \
filtering by it.

OUTPUT FORMAT — this is strict:
Return a JSON array and nothing else. No prose before or after it. No markdown \
code fences. Each element must be exactly:

  {"file": "<path>", "line": <int>, "severity": "<level>", \
"category": "<bug|security|performance|correctness>", "comment": "<text>"}

Rules for the fields:
- "line" MUST be one of the line numbers shown in the left margin of the diff \
you are given. Do not count lines yourself; use the numbers provided.
- Only cite lines marked with "+" — lines the author added in this pull \
request.
- "comment" should be one or two sentences, stating the problem and the fix. \
Write to the author, not about them.

If the code looks fine, return exactly: []

An empty array is a perfectly good answer and is the correct answer most of \
the time. Do not invent problems to appear useful."""


def render_hunk_for_review(hunk: Hunk) -> str:
    """Render a hunk with explicit line numbers in the margin.

    The model is shown the numbers rather than asked to derive them. Counting
    lines in a diff — where added, removed, and context lines advance different
    counters — is exactly the kind of arithmetic language models get slightly
    wrong, and a comment on the wrong line is worse than no comment.

    Removed lines are shown (they are context for what changed) but carry no
    line number, since they don't exist in the new version and can't be
    commented on.
    """
    out = [f"@@ {hunk.heading.strip()}" if hunk.heading.strip() else "@@"]

    for line in hunk.lines:
        if line.kind is LineKind.ADDED:
            out.append(f"{line.new_line:>6} + {line.content}")
        elif line.kind is LineKind.REMOVED:
            out.append(f"{'':>6} - {line.content}")
        else:
            out.append(f"{line.new_line:>6}   {line.content}")

    return "\n".join(out)


def render_file_for_review(file: FileDiff, hunks: list[Hunk] | None = None) -> str:
    """Render one file's changes, optionally only selected hunks."""
    chosen = hunks if hunks is not None else file.hunks
    body = "\n\n".join(render_hunk_for_review(h) for h in chosen)

    header = f"FILE: {file.path}"
    if file.status.value != "modified":
        header += f"  ({file.status.value})"

    return f"{header}\n{body}"


def build_review_prompt(rendered_files: list[str], focus: str | None = None) -> str:
    """Assemble the user message for one LLM call."""
    parts = [
        "Review the following changes from a pull request.",
        "",
        "Lines beginning with '+' were added by this pull request and are the "
        "only lines you may comment on. The number in the left margin is the "
        "line number to cite.",
        "",
    ]

    if focus:
        parts += [f"The repository maintainers asked reviewers to focus on: {focus}", ""]

    parts.append("\n\n---\n\n".join(rendered_files))
    parts += [
        "",
        "---",
        "",
        "Return your findings as a JSON array. Return [] if nothing is worth "
        "raising.",
    ]
    return "\n".join(parts)


SUMMARY_SYSTEM_PROMPT = """You are summarising an automated code review for \
the pull request author. Be brief and factual. Two or three sentences at most. \
Lead with whether anything needs attention before merge. Do not repeat the \
individual comments — they are already posted inline. Do not use headings."""


def build_summary_prompt(findings_by_severity: dict[str, int], files: int) -> str:
    if not findings_by_severity:
        return (
            f"An automated review of {files} changed file(s) found nothing "
            "worth flagging. Write a one-sentence note saying so."
        )

    breakdown = ", ".join(
        f"{count} {sev}" for sev, count in findings_by_severity.items() if count
    )
    return (
        f"An automated review of {files} changed file(s) produced: {breakdown}. "
        "Write a brief summary for the author."
    )
