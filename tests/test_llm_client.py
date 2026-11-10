"""The real Anthropic client, exercised end to end against a fake HTTP server.

Every other test stubs the model out, which is why a dependency bump once
broke client construction without a single test failing. This one builds the
real SDK client and sends a real request through httpx, so an incompatible
anthropic/httpx pair fails CI instead of the first live review.
"""

import json

import httpx

from app.llm import ClaudeClient


def _fake_messages_api(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/messages"
    body = json.loads(request.content)
    assert body["system"] == "review this"
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant",
        "model": body["model"], "stop_reason": "end_turn", "stop_sequence": None,
        "content": [{"type": "text", "text": "[]"}],
        "usage": {"input_tokens": 120, "output_tokens": 2},
    })


async def test_real_client_constructs_and_round_trips():
    from anthropic import AsyncAnthropic

    client = ClaudeClient(api_key="test-key")
    client._client = AsyncAnthropic(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_fake_messages_api)),
    )

    text, tokens_in, tokens_out = await client.complete("review this", "diff")

    assert text == "[]"
    assert (tokens_in, tokens_out) == (120, 2)
