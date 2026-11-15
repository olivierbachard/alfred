import dataclasses
from collections.abc import Iterator

import httpx
from pydantic import ValidationError

from alfred.llm.base import ChatResponse, LLMError, Message, StreamChunk
from alfred.llm.openai_compat.models import STREAM_DONE_TOKEN, OpenaiChatResponse


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.http_client = http_client or httpx.Client()

    def chat(self, messages: list[Message]) -> ChatResponse:

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self.http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [dataclasses.asdict(msg) for msg in messages],
                },
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"{e.request.url} returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise LLMError(f"Could not reach {e.request.url}: {e}") from e

        try:
            response_json = OpenaiChatResponse.model_validate_json(response.content)
            choice = response_json.choices[0] if response_json.choices else None
            usage = response_json.usage if response_json.usage else None
            return ChatResponse(
                content=choice.message.content if choice and choice.message else "",
                finish_reason=choice.finish_reason if choice else None,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
            )
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response from {response.url}: {e!r}") from e

    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with self.http_client.stream(
                method="POST",
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [dataclasses.asdict(msg) for msg in messages],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                # No read limit, since generation can be slow; bound the connection step.
                timeout=httpx.Timeout(None, connect=5.0),
            ) as response:
                if response.is_error:
                    # A streamed body is unread until asked for; the error message needs it.
                    response.read()
                response.raise_for_status()
                finished = False
                for line in response.iter_lines():
                    line_data = line.removeprefix("data:").strip()

                    if not line_data:
                        continue

                    if line_data == STREAM_DONE_TOKEN:
                        finished = True
                        yield StreamChunk(
                            content="",
                            done=True,
                            finish_reason=None,
                            input_tokens=None,
                            output_tokens=None,
                        )
                        break

                    line_json = OpenaiChatResponse.model_validate_json(line_data)
                    choice = line_json.choices[0] if line_json.choices else None
                    usage = line_json.usage if line_json.usage else None
                    yield StreamChunk(
                        content=choice.delta.content
                        if choice and choice.delta and choice.delta.content
                        else "",
                        done=False,
                        finish_reason=choice.finish_reason if choice else None,
                        input_tokens=usage.prompt_tokens if usage else None,
                        output_tokens=usage.completion_tokens if usage else None,
                    )

                if not finished:
                    raise LLMError("Stream interrupted before done")
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"{e.request.url} returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise LLMError(f"Could not reach {e.request.url}: {e}") from e
        except ValidationError as e:
            raise LLMError(f"Cannot parse a stream chunk: {e}") from e
