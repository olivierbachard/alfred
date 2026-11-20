import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROVIDERS = ("openai_compat", "anthropic")


class ConfigError(Exception):
    pass


@dataclass
class AppConfig:
    base_url: str
    model: str
    provider: str
    api_key: str | None
    anthropic_api_key: str | None


@cache
def get_config() -> AppConfig:

    config_path = Path(__file__).parent.parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    provider = config["llm"].get("provider")
    if provider is None:
        raise ConfigError(
            f"Missing 'provider' in [llm] of {config_path}. "
            f"Expected one of: {', '.join(PROVIDERS)}."
        )
    if provider not in PROVIDERS:
        raise ConfigError(
            f"Unknown provider {provider!r} in [llm] of {config_path}. "
            f"Expected one of: {', '.join(PROVIDERS)}."
        )

    return AppConfig(
        base_url=config["llm"]["base_url"],
        model=config["llm"]["model"],
        provider=provider,
        api_key=os.getenv("API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
