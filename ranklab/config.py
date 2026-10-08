"""Settings: environment variables plus an optional .env file in the working directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# provider name -> (base-URL env var, default base URL, API-key env var or None)
PROVIDERS: dict[str, tuple[str, str, str | None]] = {
    "openai": ("OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434/v1", None),
}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str | None


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file. Variables already set in the environment win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def provider_config(name: str) -> ProviderConfig:
    if name not in PROVIDERS:
        known = ", ".join(sorted([*PROVIDERS, "mock"]))
        raise ValueError(f"unknown provider {name!r}; use one of: {known}")
    url_var, default_url, key_var = PROVIDERS[name]
    api_key = os.environ.get(key_var) if key_var else None
    return ProviderConfig(name, os.environ.get(url_var, default_url).rstrip("/"), api_key or None)


def env_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]
