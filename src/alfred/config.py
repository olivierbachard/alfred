import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    base_url: str
    model: str
    api_key: str | None


@cache
def get_config() -> AppConfig:

    config_path = Path(__file__).parent.parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        return AppConfig(
            base_url=config["llm"]["base_url"],
            model=config["llm"]["model"],
            api_key=os.getenv("API_KEY"),
        )
