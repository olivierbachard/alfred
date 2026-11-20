import typer

from alfred.agent.conversation import Conversation
from alfred.config import ConfigError, get_config
from alfred.llm.anthropic.llm_provider import AnthropicProvider
from alfred.llm.base import ChatResponse, LLMError, LLMProvider, Message
from alfred.llm.openai_compat.llm_provider import OpenAICompatProvider

app = typer.Typer()

SYSTEM_PROMPT = "You are a helpful assistant named Alfred."


@app.callback()
def cli():
    """Ask an AI model questions from the command line."""


@app.command()
def ask(question: str):
    """Ask the model a single question and print the answer."""
    provider = _build_provider()

    try:
        response = _chat_with_provider(
            provider=provider,
            messages=[
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=question),
            ],
        )
    except LLMError:
        raise typer.Exit(code=1)

    typer.echo(f"Response: {response.content}")
    typer.echo(f"Finish reason: {response.finish_reason}")
    typer.echo(f"Input tokens: {response.input_tokens}")
    typer.echo(f"Output tokens: {response.output_tokens}")


def _build_provider() -> LLMProvider:
    try:
        config = get_config()
    except ConfigError as e:
        typer.echo(f"Config error: {e}", err=True)
        raise typer.Exit(code=1)

    if config.provider == "anthropic":
        if not config.anthropic_api_key:
            typer.echo("Missing ANTHROPIC_API_KEY env variable", err=True)
            raise typer.Exit(code=1)
        return AnthropicProvider(api_key=config.anthropic_api_key, model=config.model)

    return OpenAICompatProvider(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
    )


def _chat_with_provider(provider: LLMProvider, messages: list[Message]):
    try:
        return provider.chat(messages=messages)
    except LLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise


def _chat_stream_with_provider(
    provider: LLMProvider, messages: list[Message]
) -> ChatResponse:

    try:
        response: str = ""
        finish_reason: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        for chunk in provider.chat_stream(messages=messages):
            if chunk.done:
                break

            finish_reason = chunk.finish_reason if not finish_reason else finish_reason
            input_tokens = chunk.input_tokens if input_tokens is None else input_tokens
            output_tokens = (
                chunk.output_tokens if output_tokens is None else output_tokens
            )
            response += chunk.content
            typer.echo(chunk.content, nl=False)
        typer.echo()
        return ChatResponse(
            content=response,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except LLMError as e:
        if response:
            typer.echo()
        typer.echo(f"Error: {e}", err=True)
        raise


@app.command()
def chat():
    """Chat with the model over several turns. Type /exit to quit."""
    conversation: Conversation = Conversation(SYSTEM_PROMPT)
    provider = _build_provider()

    try:
        while True:
            message = typer.prompt("Enter your next message")
            message = message.strip()
            if not message:
                continue

            if message == "/exit":
                break

            conversation.add_user_message(message)
            try:
                response = _chat_stream_with_provider(
                    provider=provider, messages=conversation.get_messages()
                )
            except LLMError:
                conversation.remove_last_message()
                typer.echo("An error occurred, please retry in a moment.")
                continue
            except KeyboardInterrupt:
                raise typer.Abort()

            conversation.add_assistant_message(content=response.content)

            typer.echo(f"Finish reason: {response.finish_reason}")
            typer.echo(f"Input tokens: {response.input_tokens}")
            typer.echo(f"Output tokens: {response.output_tokens}")
    finally:
        typer.echo("\nGood bye! ")
