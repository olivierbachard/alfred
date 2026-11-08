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
    finish_reason: str
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    def chat(self, messages: list[Message]) -> ChatResponse: ...
