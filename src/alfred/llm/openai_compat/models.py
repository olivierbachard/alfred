from pydantic import BaseModel

STREAM_DONE_TOKEN = "[DONE]"


class PromptUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class ChoiceMessage(BaseModel):
    content: str | None = None
    role: str | None = None


class Choice(BaseModel):
    delta: ChoiceMessage | None = None
    message: ChoiceMessage | None = None
    finish_reason: str | None = None


class OpenaiChatResponse(BaseModel):
    choices: list[Choice]
    usage: PromptUsage | None = None
