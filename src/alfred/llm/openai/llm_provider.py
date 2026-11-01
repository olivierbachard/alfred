import dataclasses

import httpx
from alfred.llm.base import ChatResponse, LLMError, Message


class OpenAIProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url
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
            response_json = response.json()
            return ChatResponse(
                content=response_json["choices"][0]["message"]["content"],
                finish_reason=response_json["choices"][0]["finish_reason"],
                input_tokens=response_json["usage"]["prompt_tokens"],
                output_tokens=response_json["usage"]["completion_tokens"],
            )
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response from {response.url}: {e!r}") from e
