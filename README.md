# pr-reviewer

[![CI](https://github.com/aaravgarai17/pr-reviewer/actions/workflows/ci.yml/badge.svg)](https://github.com/aaravgarai17/pr-reviewer/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A GitHub bot that reviews pull requests with Claude and posts **inline comments
on the exact lines that need them**.

The interesting problems here aren't "call an API with a prompt". They are:
mapping a model's output back to precise positions in a unified diff, fitting a
5,000-line pull request into a context window, and stopping a language model
from leaving forty comments nobody will read.

---

## Try it without any API keys

```bash
pip install -r requirements.txt
python demo.py
```

The demo runs the **entire production pipeline** — diff parsing, chunking,
prompt building, JSON parsing, line anchoring, noise control — with only GitHub
and Claude stubbed out.

```
==============================================================================
 3. NOISE CONTROL
==============================================================================
  the model returned              8 finding(s)
  dropped: not in the diff        1
  dropped: below severity floor   1
  dropped: duplicates             1
  dropped: over the cap           0
  posted                          5

==============================================================================
 4. WHAT GETS POSTED
==============================================================================

  ── api/users.py:17 (diff position 4)
     🔴 critical · security
     This is a live API key committed to source control. Move it to an
     environment variable and rotate the exposed key.

  ── api/users.py:21 (diff position 8)
     🟠 major · bug
     find_user() can return None, so accessing .email will raise
     AttributeError. Guard the lookup before dereferencing.
```

`python demo.py --show-prompt` additionally prints the exact prompt the model
receives.

And to check every claim in this README mechanically:

```bash
./verify.sh          # 12 checks, all offline
```

## Architecture

```
  GitHub  ──pull_request webhook──▶  ┌──────────────────────┐
                                     │  FastAPI receiver    │
                                     │  verify HMAC         │
                                     │  return 200 at once  │──▶ background
                                     └──────────────────────┘        │
                                                                     ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  1. already reviewed this SHA?  ──yes──▶ stop (idempotency)          │
   │  2. read .reviewbot.yml from the PR head                             │
   │  3. fetch the unified diff                                           │
   │  4. parse → files → hunks → lines  (positions + line numbers)        │
   │  5. drop lockfiles, binaries, deletions, oversized files             │
   │  6. pack into chunks that fit the context budget                     │
   │  7. review chunks concurrently (bounded)  ──▶  Claude                │
   │  8. parse JSON, salvaging partial responses                          │
   │  9. anchor findings to real added lines                              │
   │ 10. threshold → deduplicate → cap                                    │
   │ 11. post ONE review with inline comments + summary                   │
   └──────────────────────────────────────────────────────────────────────┘
```

---

## The three hard problems

### 1. A line number is not a diff position

This is where most of the real engineering is, and it fails silently when you
get it wrong.

A model says *"line 42 dereferences a possibly-null pointer"*. GitHub will not
accept "line 42 of the file". It wants either:

| Addressing | Meaning |
| ---------- | ------- |
| `position` | Lines down from the **first** `@@` header in that file's diff. Counts context, additions, deletions, **and subsequent `@@` headers**. |
| `line` + `side` | Absolute line number in the file, plus `LEFT` (old) or `RIGHT` (new). |

They coincide only in the trivial case: one hunk, starting at line 1, with no
deletions — exactly what you'd construct while hand-testing.

Take a hunk header of `@@ -50,4 +51,5 @@`. Its first line is **file line 51**
but **diff position 9**:

```
      pos= 1  new= 10  context  '    a = 1'
      pos= 2  new= 11  context  '    b = 2'
      pos= 3  new=None removed  '    return a'
      pos= 4  new= 12  added    '    c = 3'
      pos= 5  new= 13  added    '    return a + c'
      pos= 6  new= 14  context  ''
      pos= 7  new= 15  context  'def spacer():'
                    ← position 8 is the second @@ header itself
      pos= 9  new= 51  context  '    x = 10'
      pos=10  new=None removed  '    return x'
      pos=11  new= 52  added    '    y = 20'
```

Comment at position 51 and GitHub rejects the entire review — the diff only has
12 positions. Comment at position 9 thinking it means line 9 and your remark
lands 42 lines from where you meant.

The parser tracks both for every line, and forgetting that **a second `@@`
header consumes a position** is its own dedicated test.

This bot posts using `line` + `side` because it's far less error-prone, but
computes `position` too — the legacy endpoint needs it, and the two together
make the distinction impossible to fudge.

### 2. Fitting a pull request into a context window

A 5,000-line PR doesn't fit. Even where it would, a single failed call
shouldn't lose the whole review. The strategy, in order:

1. **Pack whole files together** while they fit the budget — a file reviewed
   in one piece gives the model the most coherent view of it.
2. **One file alone** if it fills a request by itself.
3. **Split by hunk** if even one file is too large. Git already chose hunks as
   coherent regions of change.
4. **Skip** a lone hunk that exceeds the budget. Truncating mid-function makes
   the model confidently wrong about code it cannot see — worse than silence.

Chunks are reviewed concurrently behind a semaphore. Unbounded, a 50-file PR
would fire 50 simultaneous requests and hit the provider's rate limit, turning
a slow review into a failed one. **One failing chunk doesn't sink the rest** —
the review posts what the others found.

### 3. Stopping the bot from being annoying

An unfiltered LLM produces forty comments on a two-hundred-line PR. Every one
is individually plausible. Collectively they're unusable: the author skims,
hits a nitpick, stops reading. After two such reviews the bot gets muted — and
a muted bot has *negative* value, having spent budget teaching the team to
ignore it.

So the pipeline is deliberately lossy:

| Stage | Removes |
| ----- | ------- |
| **Anchor** | Findings citing files or lines not in the diff |
| **Threshold** | Anything below the configured severity |
| **Deduplicate** | The same point made twice, by wording similarity or shared line |
| **Cap** | All but the most severe N |

Order matters. Deduplicating before thresholding would let a `nit` survive as
the representative of a group whose `major` sibling was dropped. Capping before
sorting would keep whichever comments happened to arrive first.

Capped comments are **chosen** by severity but **presented** in file order, so
the author reads straight down their diff.

---

## The prompt, and how it got there

Each rule below exists because its absence produced a specific failure.

| Rule in the prompt | What happened without it |
| ------------------ | ------------------------ |
| *"Return `[]` if the code is fine"* | The model treated an empty response as failure and manufactured filler — "consider adding a comment here" on a two-line change. |
| *"Do not comment on formatting"* | Roughly half the output was quoting, spacing, and import order. Linters do that better, deterministically, and free. |
| Severity **definitions**, not just labels | Given bare labels, nearly everything came back `major`, making the field useless for filtering. |
| *"Cite the number in the margin"* | Asked to count lines itself, the model was off by a few — and a comment on the wrong line is worse than no comment. |
| *"JSON only, no code fences"* | Markdown fences around JSON were the single most common parse failure. |
| *"Only comment on added lines"* | Remarks on unchanged context that the author didn't write and can't act on. |

The parser strips fences and trailing commas defensively anyway — instructions
reduce failures, they don't eliminate them.

**Partial responses are salvaged.** If one finding in eight has an invalid
severity, that one is dropped and the other seven are kept. Discarding the
batch would throw away good comments over a typo.

---

## Configuration

Repositories configure the bot with a committed `.reviewbot.yml`, read from
the **head of each PR** — so a pull request that changes the config is reviewed
under its new settings.

```yaml
enabled: true
severity_threshold: minor      # critical | major | minor | nit
max_comments: 10
ignore_paths:
  - "migrations/*"
focus: "SQL injection and missing authorization checks"
post_summary: true
```

`ignore_paths` **extends** the built-in list (lockfiles, minified bundles,
vendored dependencies, generated protobufs) rather than replacing it — nobody
adding one pattern expects to re-enable review comments on `package-lock.json`.

`max_comments` is clamped to the server's hard limit, so a repository cannot
configure itself into being spammed. Invalid YAML falls back to defaults with a
warning rather than blocking the review; the PR author usually didn't write
that file and can't fix it.

## Security

The webhook endpoint verifies GitHub's HMAC signature **before parsing the
body**. Without it, anyone on the internet could make this bot review arbitrary
repositories on your API budget. Comparison uses `hmac.compare_digest`, not
`==`, because a plain comparison leaks through timing how much of a forged
signature was correct.

CI additionally greps the repository for anything resembling a committed API
key on every push.

## Running it

**Requires:** Python 3.10+. To post real reviews you also need a GitHub token
and an Anthropic API key — see `.env.example`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in your keys
uvicorn app.main:app --reload
```

**Start in dry run.** `DRY_RUN=true` computes the full review and logs exactly
what it would post, without touching the pull request. Worth doing on a few
real PRs before letting it comment.

Review an existing PR by hand, no webhook required:

```bash
curl -X POST http://localhost:8000/review/OWNER/REPO/123
```

For live webhooks, expose the port (`ngrok http 8000`) and add a webhook to
the repo pointing at `https://<your-url>/webhook`, content type
`application/json`, with the same secret as `GITHUB_WEBHOOK_SECRET`, subscribed
to **Pull requests**.

### Cost

Roughly **$0.005–$0.02 per pull request** on Sonnet, depending on size. The
demo reports an estimate for its sample. Levers, in order of effect: raise
`severity_threshold` (fewer output tokens), lower `max_file_lines`, or switch
to Haiku (~5× cheaper, noticeably blunter).

## Tests

```bash
pytest -q                                    # 161 tests
pytest --cov=app --cov-report=term-missing   # 91%
```

Everything runs offline. GitHub is mocked with `respx`; Claude is replaced by a
stub returning canned responses. The suite covers diff parsing across renames,
deletions, new files, binaries, multi-hunk files and the `\ No newline` marker;
malformed model output; HMAC verification including tampered bodies; chunking
boundaries; every filter stage; and the full pipeline end to end.

CI runs it on Python 3.10/3.11/3.12, executes the demo to prove the offline
path still works, and scans for committed credentials.

---

## What doesn't work well

- **No whole-file context.** The model sees the diff plus git's few context
  lines, not the surrounding file. It will occasionally flag something the rest
  of the file already handles. Fetching full files would improve precision and
  multiply cost; this is the main quality/cost trade-off in the project.
- **No cross-file reasoning.** A chunk boundary hides the relationship between
  a caller and callee in different files.
- **Line snapping is a heuristic.** A citation within 3 lines of a real added
  line gets moved to it. Usually right, occasionally attaches a comment to a
  neighbour.
- **Severity is the model's opinion.** Prompt thresholds constrain it but
  calibration still drifts between runs.
- **No learning from feedback.** Resolved or downvoted comments don't inform
  future reviews. That's the obvious next feature.
- **Single process, in-memory background tasks.** A restart mid-review loses
  it. A real deployment would use a durable queue.
- **English-language and mainstream-language bias.** Weaker on less common
  languages than on Python/JS/Go.

## Layout

```
pr-reviewer/
├── app/
│   ├── diff.py        # unified diff parser + position mapping  ← the core
│   ├── prompts.py     # review prompt and the reasoning behind each rule
│   ├── llm.py         # Claude client + defensive JSON parsing
│   ├── chunking.py    # context budget management
│   ├── filters.py     # anchoring, threshold, dedupe, cap
│   ├── github.py      # API client, HMAC verification, idempotency
│   ├── review.py      # the pipeline
│   ├── config.py      # server settings + .reviewbot.yml
│   ├── models.py
│   └── main.py        # FastAPI webhook receiver
├── tests/             # 161 tests, no keys or network needed
├── demo.py            # full pipeline offline, with output
├── verify.sh          # 12 checks proving this README
├── .reviewbot.yml     # example repo config
├── Dockerfile
└── docker-compose.yml
```
