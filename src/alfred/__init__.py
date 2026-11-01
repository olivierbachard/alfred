import typer

from alfred.config import get_config
from alfred.llm.base import LLMError, Message
from alfred.llm.openai_compat.llm_provider import OpenAICompatProvider

app = typer.Typer()

SYSTEM_PROMPT = "You are a helpful assistant named Alfred."


@app.callback()
def cli():
    """Ask an AI model questions from the command line."""


@app.command()
def ask(question: str):
    config = get_config()
    provider = OpenAICompatProvider(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
    )

    try:
        response = provider.chat(
            messages=[
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=question),
            ]
        )
    except LLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Response: {response.content}")
    typer.echo(f"Finish reason: {response.finish_reason}")
    typer.echo(f"Input tokens: {response.input_tokens}")
    typer.echo(f"Output tokens: {response.output_tokens}")
