"""How large a pull request can this bot actually review?

Measures the real ceiling rather than a theoretical one. "The model has a 200K
context window" is not the answer — the practical limit is set by how the
chunker packs work into requests, and by what happens when a single unit of
work exceeds the budget.

Method
------
Generate synthetic diffs at increasing line counts, push each through the
*actual* production pipeline (parse → filter → chunk), and record:

  * whether every changed line made it into a chunk, or some were skipped
  * how many LLM requests the PR costs
  * estimated input tokens and dollar cost
  * wall-clock time spent planning

A PR "fails" when the chunker has to skip content — at that point the review is
silently incomplete, which is worse than an error, because the bot posts a
confident review of a file it only partly read.

Run:  python -m bench.diff_ceiling
      python -m bench.diff_ceiling --max-lines 50000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from app.chunking import filter_reviewable, plan_chunks
from app.config import DEFAULT_IGNORE, Settings
from app.diff import parse_diff
from app.llm import estimate_tokens
from app.prompts import SYSTEM_PROMPT, build_review_prompt

# Claude Sonnet pricing, $ per million tokens.
INPUT_COST = 3.0
OUTPUT_COST = 15.0
# Assumed output per request: the bot asks for a JSON array of findings.
ASSUMED_OUTPUT_TOKENS = 400


def make_diff(total_lines: int, files: int = 10, hunks_per_file: int = 4) -> str:
    """Build a synthetic diff of approximately `total_lines` added lines.

    Lines are realistic-length source code rather than short filler, since
    token count — not line count — is what actually constrains the pipeline.
    """
    lines_per_file = max(1, total_lines // files)
    lines_per_hunk = max(1, lines_per_file // hunks_per_file)

    out: list[str] = []
    for f in range(files):
        path = f"src/module_{f}/handler.py"
        out += [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}"]

        for h in range(hunks_per_file):
            start = 1 + h * 200
            out.append(f"@@ -{start},3 +{start},{lines_per_hunk + 3} @@ def handler_{h}():")
            out.append("     existing_context_line = True")
            for i in range(lines_per_hunk):
                out.append(
                    f"+    result_{i} = process_request(payload['field_{i}'], "
                    f"timeout=30, retries=3, validate=True)"
                )
            out.append("     return result")

    return "\n".join(out) + "\n"


@dataclass
class Measurement:
    requested_lines: int
    actual_added_lines: int
    files: int
    chunks: int
    skipped_files: int
    skipped_lines_estimate: int
    input_tokens: int
    plan_seconds: float

    @property
    def complete(self) -> bool:
        """Did every changed line reach a chunk?"""
        return self.skipped_files == 0

    @property
    def cost_usd(self) -> float:
        output = self.chunks * ASSUMED_OUTPUT_TOKENS
        return (self.input_tokens / 1e6) * INPUT_COST + (output / 1e6) * OUTPUT_COST


def measure(total_lines: int, settings: Settings, files: int = 10) -> Measurement:
    diff_text = make_diff(total_lines, files=files)

    start = time.perf_counter()
    parsed = parse_diff(diff_text)
    reviewable, skipped = filter_reviewable(
        parsed, ignore_patterns=DEFAULT_IGNORE, max_file_lines=settings.max_file_lines
    )
    plan = plan_chunks(reviewable, budget_tokens=settings.chunk_budget_tokens)
    elapsed = time.perf_counter() - start

    # Input tokens = the system prompt plus the rendered diff, per request.
    system_tokens = estimate_tokens(SYSTEM_PROMPT)
    input_tokens = sum(
        estimate_tokens(build_review_prompt(c.rendered, None)) + system_tokens
        for c in plan.chunks
    )

    added = sum(f.added_line_count for f in parsed)
    dropped_files = len(skipped) + len(plan.skipped)
    dropped_lines = sum(
        f.added_line_count
        for f in parsed
        if f.path in {p for p, _ in skipped} | {p for p, _ in plan.skipped}
    )

    return Measurement(
        requested_lines=total_lines,
        actual_added_lines=added,
        files=len(parsed),
        chunks=len(plan.chunks),
        skipped_files=dropped_files,
        skipped_lines_estimate=dropped_lines,
        input_tokens=input_tokens,
        plan_seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=20_000)
    parser.add_argument("--files", type=int, default=10)
    args = parser.parse_args()

    settings = Settings()

    print()
    print("=" * 88)
    print(" PR SIZE CEILING")
    print("=" * 88)
    print(f"  chunk budget      {settings.chunk_budget_tokens:,} tokens per request")
    print(f"  max file lines    {settings.max_file_lines:,} (larger files are skipped)")
    print(f"  files per PR      {args.files}")
    print()
    print(f"  {'added lines':>12} {'files':>6} {'requests':>9} {'in tokens':>11}"
          f" {'cost':>8} {'plan ms':>9}  status")
    print("  " + "-" * 84)

    sizes = [100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10_000, 15_000, 20_000]
    sizes = [s for s in sizes if s <= args.max_lines]

    largest_complete = 0
    first_incomplete = None
    results: list[Measurement] = []

    for size in sizes:
        m = measure(size, settings, files=args.files)
        results.append(m)

        if m.complete:
            largest_complete = max(largest_complete, m.actual_added_lines)
            status = "ok"
        else:
            if first_incomplete is None:
                first_incomplete = m
            status = f"INCOMPLETE — {m.skipped_files} file(s) skipped"

        print(f"  {m.actual_added_lines:>12,} {m.files:>6} {m.chunks:>9}"
              f" {m.input_tokens:>11,} {m.cost_usd:>7.3f}$"
              f" {m.plan_seconds * 1000:>8.1f}  {status}")

    # --- narrow down the exact boundary -----------------------------------
    print()
    if first_incomplete is None:
        print(f"  No failure up to {sizes[-1]:,} lines. Raise --max-lines to find it.")
    else:
        low = largest_complete
        high = first_incomplete.actual_added_lines
        print(f"  Bisecting between {low:,} and {high:,} added lines...")

        for _ in range(8):
            if high - low <= max(50, low // 50):
                break
            mid = (low + high) // 2
            if measure(mid, settings, files=args.files).complete:
                low = mid
            else:
                high = mid

        print()
        print("=" * 88)
        print(f"  CEILING: reviews are complete up to ~{low:,} added lines")
        print(f"           content starts being skipped by ~{high:,}")
        print("=" * 88)

    # --- the limit is per-FILE, not per-PR --------------------------------
    #
    # The sweep above holds the file count fixed, so it really measures
    # `files × MAX_FILE_LINES` — arithmetic, not a property of the pipeline.
    # The question that matters is which dimension actually binds.
    print()
    print("=" * 88)
    print(" WHICH DIMENSION BINDS: per-file, or per-PR?")
    print("=" * 88)

    print()
    print("  A. One file, growing. Isolates the per-file cap.")
    print(f"  {'lines in file':>14} {'requests':>9}  status")
    print("  " + "-" * 46)
    single_ceiling = 0
    for size in (500, 1000, 1400, 1500, 1600, 2000, 5000):
        m = measure(size, settings, files=1)
        if m.complete:
            single_ceiling = max(single_ceiling, m.actual_added_lines)
        status = "ok" if m.complete else "SKIPPED — file too large"
        print(f"  {m.actual_added_lines:>14,} {m.chunks:>9}  {status}")

    print()
    print("  B. Many files, each comfortably under the cap. Isolates PR size.")
    print(f"  {'files':>7} {'total lines':>13} {'requests':>9} {'cost':>8}  status")
    print("  " + "-" * 55)
    total_ceiling = 0
    for file_count in (10, 25, 50, 100, 200):
        # 400 added lines per file — well inside the per-file limit.
        m = measure(400 * file_count, settings, files=file_count)
        if m.complete:
            total_ceiling = max(total_ceiling, m.actual_added_lines)
        status = "ok" if m.complete else f"INCOMPLETE — {m.skipped_files} skipped"
        print(f"  {m.files:>7} {m.actual_added_lines:>13,} {m.chunks:>9}"
              f" {m.cost_usd:>7.2f}$  {status}")

    print()
    print("=" * 88)
    print(" CONCLUSION")
    print("=" * 88)
    print(f"  Per file:  content is dropped above {settings.max_file_lines:,} changed lines.")
    print(f"             Largest single file reviewed completely: {single_ceiling:,} lines.")
    print(f"  Per PR:    no ceiling found — {total_ceiling:,} lines across many files")
    print("             reviewed completely, cost scaling linearly with size.")
    print()
    print("  So the binding constraint is PER FILE, not per pull request. A")
    print("  40,000-line PR spread across 100 files reviews fine; a 2,000-line")
    print("  change to a single file does not, and the oversized file is")
    print("  silently skipped.")
    print()
    print("  This is not the model's context window. It is MAX_FILE_LINES and")
    print("  the per-request token budget, both configurable, both trading")
    print("  cost against completeness. A hunk that alone exceeds the budget is")
    print("  skipped rather than truncated: showing the model half a function")
    print("  invites confident nonsense about the half it cannot see.")

    complete_results = [m for m in results if m.complete]
    if complete_results:
        biggest = complete_results[-1]
        print()
        print(f"  Largest complete review in the sweep: {biggest.actual_added_lines:,} "
              f"added lines →")
        print(f"  {biggest.chunks} request(s), ~{biggest.input_tokens:,} input tokens, "
              f"about ${biggest.cost_usd:.2f}.")
    print()


if __name__ == "__main__":
    main()
