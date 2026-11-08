import typer

from alfred.agent.conversation import Conversation
from alfred.config import get_config
from alfred.llm.base import LLMError, LLMProvider, Message
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
    config = get_config()
    provider = OpenAICompatProvider(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
    )
    return provider

def _chat_with_provider(provider: LLMProvider, messages: list[Message]):
    try:
        return provider.chat(messages=messages)
    except LLMError as e:
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
                response = _chat_with_provider(
                    provider=provider, messages=conversation.get_messages()
                )
            except LLMError:
                conversation.remove_last_message()
                typer.echo("An error occurred, please retry in a moment.")
                continue
            except KeyboardInterrupt:
                raise typer.Abort()

            conversation.add_assistant_message(content=response.content)

            typer.echo(f"Response: {response.content}")
            typer.echo(f"Finish reason: {response.finish_reason}")
            typer.echo(f"Input tokens: {response.input_tokens}")
            typer.echo(f"Output tokens: {response.output_tokens}")
    finally:
        typer.echo("\nGood bye! ")
