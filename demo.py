"""Run the full review pipeline offline, with no API keys and no network.

Everything except the two external boundaries is real: the diff parser, the
chunker, the prompt builder, the JSON parser, the anchoring logic, and the
noise-control pipeline all run exactly as they do in production. Only GitHub
and Claude are replaced by stubs.

That makes the interesting part inspectable — you can see the prompt the model
would receive, and see how raw findings get filtered down to what actually
gets posted.

Run:  python demo.py
      python demo.py --show-prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import textwrap

from app.config import RepoConfig, Settings
from app.diff import parse_diff
from app.filters import filter_pipeline
from app.models import PullRequestRef, Severity
from app.review import Reviewer

# A diff containing deliberate, findable problems: an unchecked None, a
# hardcoded secret, an N+1 query, and a bare except.
SAMPLE_DIFF = '''diff --git a/api/users.py b/api/users.py
index 1111111..2222222 100644
--- a/api/users.py
+++ b/api/users.py
@@ -14,6 +14,24 @@ from .db import database
 logger = logging.getLogger(__name__)


+API_KEY = "sk-live-8f2a9c4e1b7d3f6a"
+
+
+def get_user_email(user_id):
+    user = database.find_user(user_id)
+    return user.email
+
+
+def load_orders_for_users(user_ids):
+    orders = []
+    for uid in user_ids:
+        orders.append(database.query("SELECT * FROM orders WHERE user=" + str(uid)))
+    return orders
+
+
+def safe_parse(payload):
+    try:
+        return json.loads(payload)
+    except:
+        return None
+
 def healthcheck():
     return {"status": "ok"}
diff --git a/package-lock.json b/package-lock.json
index 3333333..4444444 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,3 +1,4 @@
 {
+  "lockfileVersion": 3,
   "name": "app"
 }
'''

# What a model typically returns for the diff above: a mix of genuinely useful
# findings, one duplicate, one style nit, and one citing a line outside the
# diff — so every stage of the filter pipeline has something to do.
CANNED_RESPONSE = json.dumps([
    {
        "file": "api/users.py", "line": 17, "severity": "critical",
        "category": "security",
        "comment": "This is a live API key committed to source control. Move it to an environment variable and rotate the exposed key.",
    },
    {
        "file": "api/users.py", "line": 21, "severity": "major",
        "category": "bug",
        "comment": "find_user() can return None, so accessing .email will raise AttributeError. Guard the lookup before dereferencing.",
    },
    {
        "file": "api/users.py", "line": 27, "severity": "critical",
        "category": "security",
        "comment": "String concatenation into SQL allows injection. Use a parameterised query instead.",
    },
    {
        "file": "api/users.py", "line": 26, "severity": "major",
        "category": "performance",
        "comment": "This issues one query per user (N+1). Fetch all orders in a single query with WHERE user IN (...).",
    },
    {
        "file": "api/users.py", "line": 33, "severity": "minor",
        "category": "correctness",
        "comment": "A bare except swallows KeyboardInterrupt and SystemExit. Catch json.JSONDecodeError specifically.",
    },
    {
        "file": "api/users.py", "line": 21, "severity": "major",
        "category": "bug",
        "comment": "find_user() may return None and .email will then raise AttributeError.",
    },
    {
        "file": "api/users.py", "line": 9999, "severity": "major",
        "category": "bug",
        "comment": "This line has a problem.",
    },
    {
        "file": "api/users.py", "line": 18, "severity": "nit",
        "category": "correctness",
        "comment": "Consider adding a blank line here for readability.",
    },
])


class StubGitHub:
    def __init__(self):
        self.posted = None

    async def already_reviewed(self, pr):
        return False

    async def get_repo_config(self, pr, path=".reviewbot.yml"):
        return None

    async def get_diff(self, pr):
        return SAMPLE_DIFF

    async def post_review(self, pr, comments, summary, event="COMMENT"):
        self.posted = {"comments": comments, "summary": summary}
        return {"id": 1}

    async def post_issue_comment(self, pr, body):
        self.posted = {"comments": [], "summary": body}
        return {"id": 2}


class StubClaude:
    def __init__(self, show_prompt: bool = False):
        self.show_prompt = show_prompt

    async def complete(self, system, user, max_tokens=4096):
        if self.show_prompt:
            print(banner("PROMPT SENT TO THE MODEL"))
            print(user)
            print()
        return CANNED_RESPONSE, 1450, 320


def banner(text: str) -> str:
    return f"\n{'=' * 78}\n {text}\n{'=' * 78}"


async def main(show_prompt: bool) -> None:
    settings = Settings(github_token="stub", anthropic_api_key="stub")
    github = StubGitHub()
    reviewer = Reviewer(github, StubClaude(show_prompt), settings)
    pr = PullRequestRef(owner="octocat", repo="demo", number=42, head_sha="a1b2c3d")

    # ---- what the parser sees --------------------------------------------
    files = parse_diff(SAMPLE_DIFF)
    print(banner("1. PARSED DIFF"))
    for f in files:
        print(f"  {f.path:<28} {f.status.value:<10} "
              f"{f.added_line_count} added line(s)")

    print(banner("2. REVIEWING"))
    result = await reviewer.review_pull_request(pr)

    # ---- how noise control narrowed it down -------------------------------
    reviewable = {f.path: f for f in files if f.path.endswith(".py")}
    _, stats = filter_pipeline(
        __import__("app.llm", fromlist=["parse_findings"]).parse_findings(CANNED_RESPONSE)[0],
        reviewable,
        threshold=Severity.MINOR,
        max_comments=10,
    )

    print(banner("3. NOISE CONTROL"))
    print(f"  the model returned            {stats['raw']:>3} finding(s)")
    print(f"  dropped: not in the diff      {stats['unanchorable']:>3}")
    print(f"  dropped: below severity floor {stats['below_threshold']:>3}")
    print(f"  dropped: duplicates           {stats['duplicates']:>3}")
    print(f"  dropped: over the cap         {stats['over_cap']:>3}")
    print(f"  {'posted':<29} {stats['kept']:>3}")

    # ---- what would actually appear on the PR -----------------------------
    print(banner("4. WHAT GETS POSTED"))
    posted = github.posted or {}
    for comment in posted.get("comments", []):
        print(f"\n  ── {comment.path}:{comment.line} "
              f"(diff position {comment.position})")
        for line in comment.body.splitlines():
            if line.strip():
                print(textwrap.fill(line, 74, initial_indent="     ",
                                    subsequent_indent="     "))

    print(banner("5. SUMMARY COMMENT"))
    for line in (posted.get("summary") or "").splitlines():
        print(f"  {line}")

    print(banner("6. COST"))
    print(f"  input tokens   {result.input_tokens:>6,}")
    print(f"  output tokens  {result.output_tokens:>6,}")
    print(f"  estimated      ${result.estimated_cost_usd:.4f} for this review")
    print()
    print("  No API keys were used. GitHub and Claude were stubbed; every other")
    print("  stage above is the real production code path.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-prompt", action="store_true",
        help="print the full prompt sent to the model",
    )
    args = parser.parse_args()
    asyncio.run(main(args.show_prompt))
