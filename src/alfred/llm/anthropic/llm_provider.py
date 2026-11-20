from collections.abc import Iterator

import anthropic
import httpx2
from anthropic import Anthropic, omit
from anthropic.types import (
    AssistantMessageParam,
    MessageParam,
    SystemMessageParam,
    UserMessageParam,
)

from alfred.llm.base import ChatResponse, LLMError, Message, MessageRole, StreamChunk


class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None, max_tokens: int = 1024):
        self._model = model
        self._client = Anthropic(api_key=api_key)
        self._max_tokens = max_tokens

    def chat(self, messages: list[Message]) -> ChatResponse:
        try:
            system_message, input_messages = self._parse_messages(messages)

            rsp = self._client.messages.create(
                max_tokens=self._max_tokens,
                messages=input_messages,
                model=self._model,
                system=system_message if system_message else omit,
            )

            rsp_text = "".join(b.text for b in rsp.content if b.type == "text")

            return ChatResponse(
                content=rsp_text,
                input_tokens=rsp.usage.input_tokens,
                output_tokens=rsp.usage.output_tokens,
                finish_reason=rsp.stop_reason,
            )
        except anthropic.APIConnectionError as e:
            raise LLMError(f"The server could not be reached: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(
                f"{e.request.url} returned {e.response.status_code}: {e.response.text}"
            ) from e

    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        try:
            system_message, input_messages = self._parse_messages(messages)

            with self._client.messages.create(
                max_tokens=self._max_tokens,
                messages=input_messages,
                model=self._model,
                system=system_message if system_message else omit,
                stream=True,
            ) as stream:
                finished = False
                for event in stream:
                    if event.type == "message_start":
                        yield StreamChunk(
                            content="",
                            done=False,
                            finish_reason=None,
                            input_tokens=event.message.usage.input_tokens,
                            output_tokens=None,
                        )
                        continue

                    if event.type == "message_stop":
                        finished = True
                        yield StreamChunk(
                            content="",
                            done=True,
                            finish_reason=None,
                            input_tokens=None,
                            output_tokens=None,
                        )
                        break

                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        yield StreamChunk(
                            content=event.delta.text,
                            done=False,
                            finish_reason=None,
                            input_tokens=None,
                            output_tokens=None,
                        )
                        continue

                    if (
                        event.type == "message_delta"
                        and event.delta.stop_reason is not None
                    ):
                        yield StreamChunk(
                            content="",
                            done=False,
                            finish_reason=event.delta.stop_reason,
                            input_tokens=event.usage.input_tokens
                            if event.usage
                            else None,
                            output_tokens=event.usage.output_tokens
                            if event.usage
                            else None,
                        )
                        continue

            if not finished:
                raise LLMError("Stream interrupted before done")

        except anthropic.APIConnectionError as e:
            raise LLMError(f"The server could not be reached: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(
                f"{e.request.url} returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx2.TransportError as e:
            raise LLMError(f"The connection was lost during the answer: {e}") from e

    def _parse_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[MessageParam]]:
        system_message = next(
            (m for m in messages if m.role == MessageRole.SYSTEM), None
        )

        return (
            system_message.content if system_message else None,
            [
                self._to_message_param(message)
                for message in messages
                if message.role != MessageRole.SYSTEM
            ],
        )

    def _to_message_param(self, message: Message) -> MessageParam:
        if message.role == MessageRole.USER:
            return UserMessageParam(content=message.content, role="user")
        elif message.role == MessageRole.ASSISTANT:
            return AssistantMessageParam(content=message.content, role="assistant")
        elif message.role == MessageRole.SYSTEM:
            return SystemMessageParam(content=message.content, role="system")

        raise ValueError(f"Invalid MessageRole with value {message.role}")
