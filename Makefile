DEBUG_PORT ?= 5678
DEBUGPY = uv run --with debugpy python -m debugpy --listen $(DEBUG_PORT)
ALFRED = .venv/bin/alfred

.PHONY: run chat test fmt debug-chat debug-ask

run:
	uv run alfred

chat:
	uv run alfred chat

test:
	uv run pytest

# Sort imports, then format.
fmt:
	uv run ruff check --select I --fix .
	uv run ruff format .

# Start `alfred chat` with debugpy listening. In VS Code, run "Attach to alfred".
debug-chat:
	$(DEBUGPY) $(ALFRED) chat

# Usage: make debug-ask MSG="What is the capital of France?"
debug-ask:
	@test -n "$(MSG)" || (echo 'Usage: make debug-ask MSG="your question"' >&2; exit 1)
	$(DEBUGPY) --wait-for-client $(ALFRED) ask "$(MSG)"
