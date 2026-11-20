from enum import StrEnum

from pydantic import BaseModel


class MessageParam(BaseModel):
    content: str
    role: str


class ThinkingConfigType(StrEnum):
    enabled = "enabled"
    disabled = "disabled"
    adaptive = "adaptive"
    between_tools = "between_tools"


class ThinkingConfigParam:
    type: ThinkingConfigType


class MessagesPayload(BaseModel):
    max_tokens: int
    messages: MessageParam
    model: str
    stream: bool = False
    system: str | None = None
    thinking: ThinkingConfigParam | None = None


class MessagesResponse(BaseModel):
    id: str
    type: str
