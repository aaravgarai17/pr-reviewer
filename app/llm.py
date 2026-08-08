"""Claude client, and the defensive parsing that makes structured output usable.

The reliability problem
-----------------------
Asking a language model for JSON gets you JSON *almost* always. The residual
failures are predictable and worth handling rather than crashing on:

  * wrapped in ```json fences despite instructions not to
  * a sentence of preamble before the array
  * a trailing comma, which is valid to a human and invalid to `json.loads`
  * one malformed object inside an otherwise fine array

The last case is the important one. Discarding an entire response because one
of eight findings has a bad severity string throws away seven good comments.
This module salvages what it can and reports what it dropped.

Retries are reserved for output that is unparseable in its entirety — one extra
call, then give up. Retrying forever on a model that is reliably producing
garbage just burns money.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from pydantic import ValidationError

from app.models import Finding

log = logging.getLogger("prbot.llm")

# ```json ... ```  or  ``` ... ```
FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass
class LLMResponse:
    findings: list[Finding] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    malformed_items: int = 0
    raw_text: str = ""


class LLMClient(Protocol):
    """Interface the reviewer depends on.

    Defining it as a Protocol keeps the review pipeline testable without a
    network call or an API key: tests inject a stub that returns canned
    responses, and the production client is only exercised at the boundary.
    """

    async def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""
        ...


class ClaudeClient:
    """Thin wrapper over the Anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        # Imported lazily so the package can be imported, and the whole test
        # suite can run, without the SDK or an API key present.
        from anthropic import AsyncAnthropic

        self.model = model
        self._client = AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    async def complete(
        self, system: str, user: str, max_tokens: int = 4096
    ) -> tuple[str, int, int]:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return text, message.usage.input_tokens, message.usage.output_tokens


# ------------------------------------------------------------------ parsing


def strip_fences(text: str) -> str:
    """Remove markdown code fences if the model added them anyway."""
    match = FENCE.match(text)
    return match.group(1) if match else text.strip()


def extract_json_array(text: str) -> Optional[str]:
    """Pull the outermost JSON array out of a response.

    Handles the common case of a sentence of preamble before the array. Scans
    with bracket depth rather than a regex so that arrays nested inside objects
    don't terminate the match early, and ignores brackets inside string
    literals.
    """
    text = strip_fences(text)

    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _drop_trailing_commas(text: str) -> str:
    """`[{...},]` is invalid JSON but a very common model slip."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def parse_findings(text: str) -> tuple[list[Finding], int]:
    """Parse a model response into findings.

    Returns `(findings, malformed_count)`. Individual bad entries are skipped
    rather than failing the batch — losing one finding is much better than
    losing the other seven alongside it.
    """
    payload = extract_json_array(text)
    if payload is None:
        log.warning("no JSON array found in response: %s", text[:200])
        return [], 0

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        try:
            data = json.loads(_drop_trailing_commas(payload))
        except json.JSONDecodeError as exc:
            log.warning("could not parse JSON array: %s", exc)
            return [], 0

    if not isinstance(data, list):
        return [], 0

    findings: list[Finding] = []
    malformed = 0

    for item in data:
        if not isinstance(item, dict):
            malformed += 1
            continue
        try:
            findings.append(Finding(**item))
        except (ValidationError, TypeError) as exc:
            malformed += 1
            log.debug("dropping malformed finding %r: %s", item, exc)

    return findings, malformed


async def request_findings(
    client: LLMClient,
    system: str,
    user: str,
    max_tokens: int = 4096,
    retry_on_unparseable: bool = True,
) -> LLMResponse:
    """One review call, with a single retry if nothing parseable came back.

    The retry appends a blunt reminder about the format. It fires only when the
    response yielded no findings *and* no recognisable array — an intentional
    `[]` is a valid answer and must not trigger a retry, or every clean file
    would cost two calls.
    """
    text, in_tok, out_tok = await client.complete(system, user, max_tokens)
    findings, malformed = parse_findings(text)

    unparseable = extract_json_array(text) is None
    if unparseable and retry_on_unparseable:
        log.info("unparseable response, retrying once with a format reminder")
        retry_user = (
            user
            + "\n\nIMPORTANT: your previous response could not be parsed. "
            "Respond with a JSON array only — no prose, no markdown fences. "
            "If there is nothing to report, respond with exactly: []"
        )
        text2, in2, out2 = await client.complete(system, retry_user, max_tokens)
        findings, malformed = parse_findings(text2)
        return LLMResponse(
            findings=findings,
            input_tokens=in_tok + in2,
            output_tokens=out_tok + out2,
            malformed_items=malformed,
            raw_text=text2,
        )

    return LLMResponse(
        findings=findings,
        input_tokens=in_tok,
        output_tokens=out_tok,
        malformed_items=malformed,
        raw_text=text,
    )


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for budgeting, without a tokenizer dependency.

    Roughly four characters per token holds well enough for source code. This
    is used to decide how much to put in a request, so erring slightly high is
    the safe direction — an over-estimate splits a chunk unnecessarily, an
    under-estimate overflows the context window and fails the call.
    """
    return len(text) // 3
