from unittest import mock

import anthropic
import httpx2
import pytest
from anthropic.resources.messages import Messages
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta as RawMessageDelta

from alfred.llm.anthropic.llm_provider import AnthropicProvider
from alfred.llm.base import LLMError, Message, MessageRole


def make_rsp_message(rsp_text: str) -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_1",
        type="message",
        role="assistant",
        model="m",
        content=[TextBlock(type="text", text=rsp_text)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


@pytest.fixture
def create_rsp_messages():
    with mock.patch.object(Messages, "create", autospec=True) as m:
        m.return_value = make_rsp_message("Paris")
        yield m


def build_sut(api_key: str | None, max_tokens: int | None = None) -> AnthropicProvider:
    return AnthropicProvider(
        model="claude-haiku-5-5", api_key=api_key, max_tokens=max_tokens
    )


def make_valid_input_messages():
    return [
        Message(
            role=MessageRole.SYSTEM,
            content="you are an helpful assistant named Alfred",
        ),
        Message(role=MessageRole.USER, content="What is the capital of France"),
    ]


@mock.patch("alfred.llm.anthropic.llm_provider.Anthropic")
def test_provider_passes_api_key_to_sdk(anthropic_cls):
    build_sut("my_api_key", max_tokens=2048)

    anthropic_cls.assert_called_once_with(api_key="my_api_key")


class TestChat:
    def test_chat(self, create_rsp_messages):
        sut = build_sut("my_api_key", max_tokens=2048)
        response = sut.chat(make_valid_input_messages())
        assert create_rsp_messages.call_args.kwargs["max_tokens"] == 2048
        assert create_rsp_messages.call_args.kwargs["model"] == "claude-haiku-5-5"
        assert (
            create_rsp_messages.call_args.kwargs["system"]
            == "you are an helpful assistant named Alfred"
        )
        assert response.content == "Paris"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.finish_reason == "end_turn"

    def test_chat_catch_APIConnectionError(self, create_rsp_messages):
        req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        create_rsp_messages.side_effect = anthropic.APIConnectionError(request=req)
        sut = build_sut("my_api_key")
        with pytest.raises(LLMError, match="The server could not be reached"):
            sut.chat(make_valid_input_messages())

    def test_chat_catch_APIStatusError(self, create_rsp_messages):
        req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx2.Response(500, json={"error": "boom"}, request=req)
        create_rsp_messages.side_effect = anthropic.APIStatusError(
            "boom", response=resp, body=None
        )
        sut = build_sut("my_api_key")
        with pytest.raises(
            LLMError, match=f"{req.url} returned {resp.status_code}: {resp.text}"
        ):
            sut.chat(make_valid_input_messages())


def fake_stream(events: list) -> mock.MagicMock:
    # The real SDK stream is a context manager that yields events.
    stream = mock.MagicMock()
    stream.__enter__.return_value = iter(events)
    return stream


class TestChatStream:
    def test_chat_stream(self, create_rsp_messages):
        rsp_messages = [
            RawMessageStartEvent(
                type="message_start",
                message=AnthropicMessage(
                    id="1",
                    type="message",
                    role="assistant",
                    model="m",
                    content=[],
                    usage=Usage(input_tokens=10, output_tokens=0),
                ),
            ),
            RawContentBlockDeltaEvent(
                index=0,
                type="content_block_delta",
                delta=TextDelta(type="text_delta", text="Paris"),
            ),
            RawContentBlockDeltaEvent(
                index=0,
                type="content_block_delta",
                delta=TextDelta(type="text_delta", text="About 2.1 million people."),
            ),
            RawMessageDeltaEvent(
                type="message_delta",
                delta=RawMessageDelta(stop_reason="end_turn"),
                usage=MessageDeltaUsage(output_tokens=27),
            ),
            RawMessageStopEvent(
                type="message_stop",
            ),
        ]

        create_rsp_messages.return_value = fake_stream(rsp_messages)
        sut = build_sut("my_api_key")
        chunks = list(sut.chat_stream(make_valid_input_messages()))

        assert create_rsp_messages.call_args.kwargs["stream"] is True
        assert (
            create_rsp_messages.call_args.kwargs["system"]
            == "you are an helpful assistant named Alfred"
        )

        assert chunks[0].content == ""
        assert chunks[0].input_tokens == 10
        assert chunks[0].output_tokens is None
        assert chunks[0].finish_reason is None

        assert chunks[1].content == "Paris"
        assert chunks[1].input_tokens is None
        assert chunks[1].output_tokens is None
        assert chunks[1].finish_reason is None

        assert chunks[2].content == "About 2.1 million people."
        assert chunks[2].input_tokens is None
        assert chunks[2].output_tokens is None
        assert chunks[2].finish_reason is None

        assert chunks[3].content == ""
        assert chunks[3].input_tokens is None
        assert chunks[3].output_tokens == 27
        assert chunks[3].finish_reason == "end_turn"

    def test_chat_stream_raises_when_stream_ends_before_message_stop(
        self, create_rsp_messages
    ):
        create_rsp_messages.return_value = fake_stream(
            [
                RawContentBlockDeltaEvent(
                    index=0,
                    type="content_block_delta",
                    delta=TextDelta(type="text_delta", text="Paris"),
                ),
            ]
        )
        sut = build_sut("my_api_key")

        with pytest.raises(LLMError, match="interrupted"):
            list(sut.chat_stream(make_valid_input_messages()))

    def test_chat_stream_raises_when_connection_drops_mid_stream(
        self, create_rsp_messages
    ):
        def cut_stream():
            yield RawContentBlockDeltaEvent(
                index=0,
                type="content_block_delta",
                delta=TextDelta(type="text_delta", text="Paris"),
            )
            raise httpx2.RemoteProtocolError("peer closed connection")

        stream = mock.MagicMock()
        stream.__enter__.return_value = cut_stream()
        create_rsp_messages.return_value = stream
        sut = build_sut("my_api_key")

        with pytest.raises(LLMError, match="connection was lost"):
            list(sut.chat_stream(make_valid_input_messages()))
