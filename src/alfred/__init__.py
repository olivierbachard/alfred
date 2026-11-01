import typer

from alfred.config import get_config
from alfred.llm.base import Message
from alfred.llm.openai.llm_provider import OpenAIProvider

app = typer.Typer()

SYSTEM_PROMPT = "You are a helpful assistant."


@app.command()
def ask(question: str):
    config = get_config()
    provider = OpenAIProvider(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
    )
    response = provider.chat(
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=question),
        ]
    )

    print(f"Response: {response.content}")
    print(f"Finish reason: {response.finish_reason}")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")


def main() -> None:
    print("Hello from alfred!")
    app()
