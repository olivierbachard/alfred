import json
from pathlib import Path

import httpx
from alfred.llm.base import Message
from alfred.llm.openai.llm_provider import OpenAIProvider

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def test_chat_using_ollama():

    ollama_chat_response_data = json.loads((FIXTURE_DIR / "ollama_chat_response.json").read_text())
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/chat/completions":
            
            return httpx.Response(
                200,
                json=ollama_chat_response_data
            )
        return httpx.Response(404)
    
    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    sut = OpenAIProvider(
        base_url="http://localhost:11434",
        model="llama2",
        api_key="my_api_key",
        http_client=mock_client
    )

    provider_response = sut.chat(
        messages=[
            Message(role="user", content="Hello!")
        ]
    )

    captured_request = captured_requests[0]
    assert captured_request.url.path == "/chat/completions"
    assert json.loads(captured_request.content) == {
        "model": "llama2",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }    
        ]
    }
    
    assert captured_request.headers["Authorization"] == "Bearer my_api_key"

    assert provider_response.content == ollama_chat_response_data["choices"][0]["message"]["content"]
    assert provider_response.finish_reason == ollama_chat_response_data["choices"][0]["finish_reason"]
    assert provider_response.input_tokens == ollama_chat_response_data["usage"]["prompt_tokens"]
    assert provider_response.output_tokens == ollama_chat_response_data["usage"]["completion_tokens"]

#def test_chat_using_ollama_with_system_prompt():