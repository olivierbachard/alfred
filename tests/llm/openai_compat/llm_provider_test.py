import json
from pathlib import Path

import httpx
import pytest

from alfred.agent.conversation import Conversation
from alfred.llm.base import LLMError, Message
from alfred.llm.openai_compat.llm_provider import OpenAICompatProvider

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def build_sut(http_client: httpx.Client, api_key: str | None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url="http://localhost:11434",
        model="llama2",
        api_key=api_key,
        http_client=http_client,
    )


def test_chat_using_ollama():

    ollama_chat_response_data = json.loads(
        (FIXTURE_DIR / "ollama_chat_response.json").read_text()
    )
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/chat/completions":
            return httpx.Response(200, json=ollama_chat_response_data)
        return httpx.Response(404)

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    sut = build_sut(http_client=mock_client, api_key="my_api_key")

    provider_response = sut.chat(messages=[Message(role="user", content="Hello!")])

    captured_request = captured_requests[0]
    assert captured_request.url.path == "/chat/completions"
    assert json.loads(captured_request.content) == {
        "model": "llama2",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    assert captured_request.headers["Authorization"] == "Bearer my_api_key"

    assert (
        provider_response.content
        == ollama_chat_response_data["choices"][0]["message"]["content"]
    )
    assert (
        provider_response.finish_reason
        == ollama_chat_response_data["choices"][0]["finish_reason"]
    )
    assert (
        provider_response.input_tokens
        == ollama_chat_response_data["usage"]["prompt_tokens"]
    )
    assert (
        provider_response.output_tokens
        == ollama_chat_response_data["usage"]["completion_tokens"]
    )


def test_chat_raising_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection error", request=request)

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    sut = build_sut(http_client=mock_client, api_key="my_api_key")

    with pytest.raises(LLMError, match="Could not reach"):
        sut.chat(messages=[Message(role="user", content="Hello!")])


def test_chat_raising_when_http_status_error():
    error_body = {"error": "Internal Server Error"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=error_body)

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    sut = build_sut(http_client=mock_client, api_key="my_api_key")

    with pytest.raises(LLMError, match=r"returned 500: .*Internal Server Error"):
        sut.chat(messages=[Message(role="user", content="Hello!")])


def test_chat_raising_when_invalid_json_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"invalid json")

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    sut = build_sut(http_client=mock_client, api_key="my_api_key")

    with pytest.raises(LLMError, match="Unexpected response"):
        sut.chat(messages=[Message(role="user", content="Hello!")])


def stream_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def sse_response(body: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=body.encode(),
        headers={"Content-Type": "text/event-stream"},
    )


def test_chat_stream_yields_deltas_then_usage_then_done():
    sse_body = (FIXTURE_DIR / "ollama_chat_stream.txt").read_text()
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return sse_response(sse_body)

    sut = build_sut(http_client=stream_client(handler), api_key=None)

    chunks = list(sut.chat_stream(messages=[Message(role="user", content="Hello!")]))

    request_body = json.loads(captured_requests[0].content)
    assert captured_requests[0].url.path == "/chat/completions"
    assert request_body["stream"] is True
    assert request_body["stream_options"] == {"include_usage": True}

    assert [c.content for c in chunks] == [
        "How",
        " can",
        " I",
        " assist",
        " you",
        " today",
        "?",
        "",
        "",
        "",
    ]
    assert [c.done for c in chunks] == [False] * 9 + [True]
    assert [c.finish_reason for c in chunks if c.finish_reason] == ["stop"]

    usage_chunks = [c for c in chunks if c.input_tokens is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].input_tokens == 27
    assert usage_chunks[0].output_tokens == 8


def test_chat_stream_raises_when_stream_ends_before_done():
    sse_body = (FIXTURE_DIR / "ollama_chat_stream.txt").read_text()
    truncated_body = "\n\n".join(sse_body.split("\n\n")[:2]) + "\n\n"

    sut = build_sut(
        http_client=stream_client(lambda request: sse_response(truncated_body)),
        api_key=None,
    )

    with pytest.raises(LLMError, match="interrupted"):
        list(sut.chat_stream(messages=[Message(role="user", content="Hello!")]))


def test_chat_stream_raises_when_http_status_error():
    sut = build_sut(
        http_client=stream_client(
            lambda request: httpx.Response(500, json={"error": "boom"})
        ),
        api_key=None,
    )

    with pytest.raises(LLMError, match=r"returned 500: .*boom"):
        list(sut.chat_stream(messages=[Message(role="user", content="Hello!")]))


def test_chat_stream_raises_when_connection_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection error", request=request)

    sut = build_sut(http_client=stream_client(handler), api_key=None)

    with pytest.raises(LLMError, match="Could not reach"):
        list(sut.chat_stream(messages=[Message(role="user", content="Hello!")]))


def test_chat_stream_raises_when_chunk_is_not_valid():
    sut = build_sut(
        http_client=stream_client(
            lambda request: sse_response("data: not json\n\ndata: [DONE]\n\n")
        ),
        api_key=None,
    )

    with pytest.raises(LLMError, match="Cannot parse"):
        list(sut.chat_stream(messages=[Message(role="user", content="Hello!")]))


def test_chat_resends_full_history_on_second_turn():
    answers = ["Paris", "About 2.1 million people."]
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        answer = answers[len(captured_requests)]
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    sut = build_sut(http_client=mock_client, api_key=None)
    conversation = Conversation("You are Alfred.")

    conversation.add_user_message("What is the capital of France?")
    first_response = sut.chat(messages=conversation.get_messages())
    conversation.add_assistant_message(first_response.content)

    conversation.add_user_message("What's its population?")
    sut.chat(messages=conversation.get_messages())

    assert len(captured_requests) == 2
    assert json.loads(captured_requests[1].content)["messages"] == [
        {"role": "system", "content": "You are Alfred."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "What's its population?"},
    ]
