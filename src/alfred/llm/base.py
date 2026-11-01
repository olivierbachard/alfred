from dataclasses import dataclass
from typing import Protocol


@dataclass
class Message:
    role: str
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
    def chat(self, messages: list[Message]) -> ChatResponse:
        ...