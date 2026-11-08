from typer.testing import CliRunner

from alfred.channels import cli
from alfred.llm.base import ChatResponse, LLMError, Message, MessageRole

runner = CliRunner()


class FakeProvider:
    """Fails on the first call, then answers. Records the messages it receives."""

    def __init__(self) -> None:
        self.received: list[list[Message]] = []

    def chat(self, messages: list[Message]) -> ChatResponse:
        self.received.append(messages)
        if len(self.received) == 1:
            raise LLMError("server unreachable")
        return ChatResponse(
            content="Paris",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5,
        )


def test_chat_drops_unanswered_user_message_after_llm_error(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(cli, "_build_provider", lambda: provider)

    result = runner.invoke(
        cli.app, ["chat"], input="first question\nsecond question\n/exit\n"
    )

    assert result.exit_code == 0
    assert len(provider.received) == 2
    assert provider.received[1] == [
        Message(role=MessageRole.SYSTEM, content=cli.SYSTEM_PROMPT),
        Message(role=MessageRole.USER, content="second question"),
    ]
    assert "Paris" in result.output
