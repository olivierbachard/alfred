# Alfred

Alfred is a personal AI assistant. It's meant to read and send email, track a
stock portfolio, run spaced-repetition reviews, and track projects and courses,
reachable from Telegram and later WhatsApp.

Today, Alfred sends one question to an OpenAI-compatible model and prints the
answer with its token counts.

## Requirements

You need these tools installed:

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), or any other OpenAI-compatible endpoint

## Set up

1. Install the dependencies:

   ```sh
   uv sync
   ```

2. Copy the secrets template and fill in `API_KEY`. Ollama ignores the key, so
   any value works locally.

   ```sh
   cp .env.example .env
   ```

3. In `config.toml`, set `base_url` and `model` under `[llm]`. The defaults
   point to a local Ollama server.

4. If you use Ollama, pull the model:

   ```sh
   ollama pull llama3.2
   ```

## Run

Ask Alfred a question:

```sh
uv run alfred "What is the capital of France?"
```

## Test

Run the tests. They never call a real model.

```sh
uv run pytest
```

## Configuration

Alfred reads its settings from two files:

| File | Contents | Committed |
|---|---|---|
| `config.toml` | Non-secret settings, such as the model URL and name. | Yes |
| `.env` | Secrets, such as API keys. | No |
