from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str


@dataclass
class ChatResponse:
    content: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class StreamChunk:
    content: str
    done: bool
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    def chat(self, messages: list[Message]) -> ChatResponse: ...

    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]: ...
