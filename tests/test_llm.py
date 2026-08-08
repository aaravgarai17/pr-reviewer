"""Structured-output parsing — the defensive layer around model responses."""

import pytest

from app.llm import (
    LLMResponse,
    estimate_tokens,
    extract_json_array,
    parse_findings,
    request_findings,
    strip_fences,
)
from app.models import Severity

GOOD = """[
  {"file": "a.py", "line": 12, "severity": "major", "category": "bug",
   "comment": "This dereferences user before the None check above it."}
]"""


class StubLLM:
    """Returns canned responses; records what it was asked."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system, user, max_tokens=4096):
        self.calls.append((system, user))
        text = self.responses.pop(0) if self.responses else "[]"
        return text, 100, 50


# ------------------------------------------------------------ fence stripping


def test_strip_fences_json():
    assert strip_fences('```json\n[{"a": 1}]\n```') == '[{"a": 1}]'


def test_strip_fences_plain():
    assert strip_fences("```\n[]\n```") == "[]"


def test_strip_fences_leaves_unfenced_text():
    assert strip_fences("[]") == "[]"


# --------------------------------------------------------------- extraction


def test_extract_simple_array():
    assert extract_json_array('[{"a": 1}]') == '[{"a": 1}]'


def test_extract_ignores_preamble():
    """Models often narrate before answering, despite instructions."""
    text = 'Here are my findings:\n\n[{"a": 1}]'
    assert extract_json_array(text) == '[{"a": 1}]'


def test_extract_ignores_trailing_prose():
    text = '[{"a": 1}]\n\nLet me know if you need more detail.'
    assert extract_json_array(text) == '[{"a": 1}]'


def test_extract_handles_nested_arrays():
    """Bracket depth, not a regex — a nested array must not end the match."""
    text = '[{"a": [1, 2, 3]}, {"b": [4]}]'
    assert extract_json_array(text) == text


def test_extract_ignores_brackets_inside_strings():
    """A `]` in a comment body must not terminate the array early."""
    text = '[{"comment": "check arr[0] here"}]'
    assert extract_json_array(text) == text


def test_extract_handles_escaped_quotes():
    text = r'[{"comment": "he said \"hi\" loudly"}]'
    assert extract_json_array(text) == text


def test_extract_returns_none_without_an_array():
    assert extract_json_array("I could not find any issues.") is None


def test_extract_returns_none_on_unclosed_array():
    assert extract_json_array('[{"a": 1}') is None


# ------------------------------------------------------------------ parsing


def test_parse_valid_findings():
    findings, malformed = parse_findings(GOOD)
    assert malformed == 0
    assert len(findings) == 1
    assert findings[0].file == "a.py"
    assert findings[0].line == 12
    assert findings[0].severity is Severity.MAJOR


def test_parse_empty_array_is_a_valid_answer():
    findings, malformed = parse_findings("[]")
    assert findings == [] and malformed == 0


def test_parse_tolerates_markdown_fences():
    findings, _ = parse_findings(f"```json\n{GOOD}\n```")
    assert len(findings) == 1


def test_parse_tolerates_trailing_commas():
    text = '[{"file":"a.py","line":1,"severity":"minor","comment":"x"},]'
    findings, _ = parse_findings(text)
    assert len(findings) == 1


def test_one_bad_entry_does_not_discard_the_good_ones():
    """The important resilience property.

    Failing the whole batch because one of three findings has a bad severity
    throws away two perfectly good comments.
    """
    text = """[
      {"file":"a.py","line":1,"severity":"major","comment":"real finding"},
      {"file":"b.py","line":2,"severity":"catastrophic","comment":"bad severity"},
      {"file":"c.py","line":3,"severity":"minor","comment":"another real one"}
    ]"""
    findings, malformed = parse_findings(text)

    assert len(findings) == 2
    assert malformed == 1
    assert [f.file for f in findings] == ["a.py", "c.py"]


def test_missing_required_field_is_dropped():
    text = '[{"file":"a.py","severity":"major","comment":"no line number"}]'
    findings, malformed = parse_findings(text)
    assert findings == [] and malformed == 1


def test_empty_comment_is_rejected():
    text = '[{"file":"a.py","line":1,"severity":"minor","comment":"   "}]'
    findings, malformed = parse_findings(text)
    assert findings == [] and malformed == 1


def test_non_positive_line_is_rejected():
    text = '[{"file":"a.py","line":0,"severity":"minor","comment":"x"}]'
    findings, malformed = parse_findings(text)
    assert findings == [] and malformed == 1


def test_unparseable_response_yields_nothing():
    findings, malformed = parse_findings("I don't see any problems here.")
    assert findings == [] and malformed == 0


def test_non_list_json_yields_nothing():
    findings, _ = parse_findings('{"file": "a.py"}')
    assert findings == []


# ------------------------------------------------------------ retry behaviour


async def test_retries_once_when_response_is_unparseable():
    llm = StubLLM("Sorry, I can't do that.", GOOD)
    response = await request_findings(llm, "sys", "user")

    assert len(llm.calls) == 2
    assert "could not be parsed" in llm.calls[1][1]
    assert len(response.findings) == 1


async def test_empty_array_does_not_trigger_a_retry():
    """`[]` is a correct answer — retrying it would double the cost of every
    clean file."""
    llm = StubLLM("[]")
    response = await request_findings(llm, "sys", "user")

    assert len(llm.calls) == 1
    assert response.findings == []


async def test_gives_up_after_one_retry():
    llm = StubLLM("nope", "still nope")
    response = await request_findings(llm, "sys", "user")

    assert len(llm.calls) == 2
    assert response.findings == []


async def test_retry_can_be_disabled():
    llm = StubLLM("nope")
    await request_findings(llm, "sys", "user", retry_on_unparseable=False)
    assert len(llm.calls) == 1


async def test_token_usage_accumulates_across_retry():
    llm = StubLLM("nope", GOOD)
    response = await request_findings(llm, "sys", "user")
    assert response.input_tokens == 200
    assert response.output_tokens == 100


# ------------------------------------------------------------- token estimate


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 300) > estimate_tokens("a" * 100)
