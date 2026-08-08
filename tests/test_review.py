"""The full review pipeline, end to end, against stubbed GitHub and Claude.

No API keys, no network, no cost — the entire flow from webhook payload to
posted review is exercised with fakes at the two boundaries.
"""

import json

import pytest

from app.config import Settings
from app.models import PullRequestRef, Severity
from app.review import Reviewer

DIFF = """diff --git a/service.py b/service.py
--- a/service.py
+++ b/service.py
@@ -10,6 +10,10 @@ class UserService:
     def __init__(self, db):
         self.db = db
+
+    def get_name(self, user_id):
+        user = self.db.find(user_id)
+        return user.name
"""


class FakeGitHub:
    """Records calls; returns canned data."""

    def __init__(self, diff=DIFF, repo_config=None, already_reviewed=False):
        self._diff = diff
        self._repo_config = repo_config
        self._already_reviewed = already_reviewed

        self.posted_reviews = []
        self.posted_comments = []

    async def already_reviewed(self, pr):
        return self._already_reviewed

    async def get_repo_config(self, pr, path=".reviewbot.yml"):
        return self._repo_config

    async def get_diff(self, pr):
        return self._diff

    async def get_pull_request(self, pr):
        return {"head": {"sha": pr.head_sha}}

    async def post_review(self, pr, comments, summary, event="COMMENT"):
        self.posted_reviews.append({"comments": comments, "summary": summary})
        return {"id": 1}

    async def post_issue_comment(self, pr, body):
        self.posted_comments.append(body)
        return {"id": 2}


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses) or ["[]"]
        self.prompts = []

    async def complete(self, system, user, max_tokens=4096):
        self.prompts.append(user)
        text = self.responses.pop(0) if self.responses else "[]"
        return text, 500, 100


def findings_json(*items):
    return json.dumps(list(items))


@pytest.fixture
def pr():
    return PullRequestRef(owner="octocat", repo="hello", number=1, head_sha="abc123")


@pytest.fixture
def settings():
    return Settings(
        github_token="x",
        anthropic_api_key="y",
        default_max_comments=10,
        default_severity_threshold=Severity.MINOR,
    )


# ------------------------------------------------------------- happy path


async def test_posts_a_review_with_inline_comments(pr, settings):
    github = FakeGitHub()
    llm = FakeLLM(findings_json({
        "file": "service.py", "line": 14, "severity": "major",
        "category": "bug",
        "comment": "find() may return None; this will raise AttributeError.",
    }))

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.comments_posted == 1
    assert len(github.posted_reviews) == 1

    comment = github.posted_reviews[0]["comments"][0]
    assert comment.path == "service.py"
    assert comment.line == 14
    assert "AttributeError" in comment.body


async def test_clean_review_posts_a_summary_not_a_review(pr, settings):
    """Nothing to flag should not produce an empty inline review."""
    github = FakeGitHub()
    result = await Reviewer(github, FakeLLM("[]"), settings).review_pull_request(pr)

    assert result.comments_posted == 0
    assert github.posted_reviews == []
    assert len(github.posted_comments) == 1
    assert "nothing worth flagging" in github.posted_comments[0].lower()


async def test_token_usage_is_tracked(pr, settings):
    result = await Reviewer(FakeGitHub(), FakeLLM("[]"), settings).review_pull_request(pr)
    assert result.input_tokens == 500
    assert result.output_tokens == 100
    assert result.estimated_cost_usd > 0


async def test_prompt_contains_line_numbers(pr, settings):
    """The model is shown line numbers rather than asked to count."""
    llm = FakeLLM("[]")
    await Reviewer(FakeGitHub(), llm, settings).review_pull_request(pr)

    prompt = llm.prompts[0]
    assert "service.py" in prompt
    assert "13 +" in prompt or "14 +" in prompt


# ------------------------------------------------------------- idempotency


async def test_skips_a_commit_already_reviewed(pr, settings):
    github = FakeGitHub(already_reviewed=True)
    llm = FakeLLM()

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.skipped_reason == "already reviewed at this commit"
    assert llm.prompts == []          # no LLM call, so no cost
    assert github.posted_reviews == []


# ---------------------------------------------------------- repo config


async def test_repo_can_disable_reviews(pr, settings):
    github = FakeGitHub(repo_config="enabled: false")
    llm = FakeLLM()

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.skipped_reason == "disabled by .reviewbot.yml"
    assert llm.prompts == []


async def test_repo_severity_threshold_is_applied(pr, settings):
    github = FakeGitHub(repo_config="severity_threshold: critical")
    llm = FakeLLM(findings_json(
        {"file": "service.py", "line": 14, "severity": "major", "comment": "A major issue."},
        {"file": "service.py", "line": 15, "severity": "critical", "comment": "A critical one."},
    ))

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.findings_raw == 2
    assert result.comments_posted == 1
    assert github.posted_reviews[0]["comments"][0].severity is Severity.CRITICAL


async def test_repo_comment_cap_is_applied(pr, settings):
    github = FakeGitHub(repo_config="max_comments: 1")
    llm = FakeLLM(findings_json(
        {"file": "service.py", "line": 13, "severity": "minor", "comment": "First distinct point."},
        {"file": "service.py", "line": 14, "severity": "critical", "comment": "Second unrelated matter."},
    ))

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.comments_posted == 1
    # The cap keeps the most severe.
    assert github.posted_reviews[0]["comments"][0].severity is Severity.CRITICAL


async def test_focus_is_included_in_the_prompt(pr, settings):
    github = FakeGitHub(repo_config="focus: SQL injection risks")
    llm = FakeLLM("[]")

    await Reviewer(github, llm, settings).review_pull_request(pr)
    assert "SQL injection risks" in llm.prompts[0]


async def test_ignored_paths_are_not_reviewed(pr, settings):
    github = FakeGitHub(repo_config='ignore_paths:\n  - "service.py"')
    llm = FakeLLM()

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.skipped_reason == "no reviewable files"
    assert llm.prompts == []


# -------------------------------------------------------------- resilience


async def test_model_citing_a_bad_line_does_not_break_the_review(pr, settings):
    """GitHub rejects the whole payload if any comment is unplaceable."""
    github = FakeGitHub()
    llm = FakeLLM(findings_json(
        {"file": "service.py", "line": 9999, "severity": "major", "comment": "Outside the diff."},
        {"file": "service.py", "line": 14, "severity": "major", "comment": "A real finding here."},
    ))

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.findings_raw == 2
    assert result.comments_posted == 1


async def test_unparseable_model_output_yields_no_comments(pr, settings):
    github = FakeGitHub()
    llm = FakeLLM("I couldn't review this.", "Still can't.")

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.comments_posted == 0
    assert len(github.posted_comments) == 1     # summary still posted


async def test_binary_only_pr_is_skipped(pr, settings):
    github = FakeGitHub(diff="""diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
""")
    llm = FakeLLM()

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.skipped_reason == "no reviewable files"
    assert llm.prompts == []


async def test_a_failing_chunk_does_not_lose_the_others(pr, settings):
    """One bad LLM call shouldn't discard findings from the rest."""
    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, system, user, max_tokens=4096):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("upstream 500")
            return findings_json({
                "file": "service.py", "line": 14, "severity": "major",
                "comment": "A real finding.",
            }), 100, 20

    big_diff = DIFF + DIFF.replace("service.py", "other.py")
    github = FakeGitHub(diff=big_diff)
    # Each file is ~79 tokens, so this fits one per chunk and forces two.
    settings.chunk_budget_tokens = 100

    result = await Reviewer(github, FlakyLLM(), settings).review_pull_request(pr)

    assert result.chunks_sent >= 2
    assert result.comments_posted >= 1


# ---------------------------------------------------------------- dry run


async def test_dry_run_posts_nothing(pr, settings):
    settings.dry_run = True
    github = FakeGitHub()
    llm = FakeLLM(findings_json({
        "file": "service.py", "line": 14, "severity": "major", "comment": "Found something.",
    }))

    result = await Reviewer(github, llm, settings).review_pull_request(pr)

    assert result.findings_kept == 1
    assert result.comments_posted == 0
    assert github.posted_reviews == []
    assert github.posted_comments == []


# ---------------------------------------------------------------- summary


async def test_summary_reports_the_severity_breakdown(pr, settings):
    github = FakeGitHub()
    llm = FakeLLM(findings_json(
        {"file": "service.py", "line": 13, "severity": "critical", "comment": "Critical thing here."},
        {"file": "service.py", "line": 14, "severity": "minor", "comment": "Different minor matter."},
    ))

    await Reviewer(github, llm, settings).review_pull_request(pr)

    summary = github.posted_reviews[0]["summary"]
    assert "1 critical" in summary
    assert "1 minor" in summary
    assert "before merge" in summary       # critical findings warn explicitly


async def test_summary_mentions_suppressed_findings(pr, settings):
    github = FakeGitHub(repo_config="max_comments: 1")
    llm = FakeLLM(findings_json(
        {"file": "service.py", "line": 13, "severity": "major", "comment": "First distinct point."},
        {"file": "service.py", "line": 14, "severity": "major", "comment": "Second separate concern."},
    ))

    await Reviewer(github, llm, settings).review_pull_request(pr)
    assert "suppressed" in github.posted_reviews[0]["summary"]
