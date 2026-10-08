"""Chat clients: any OpenAI-compatible endpoint, plus a deterministic mock for demos and tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import PROVIDERS, provider_config

KNOWN_PROVIDERS = (*PROVIDERS, "mock")
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    @classmethod
    def parse(cls, text: str) -> ModelSpec:
        """'ollama:llama3.2:3b' -> (ollama, llama3.2:3b). A bare model name means OpenAI."""
        text = text.strip()
        if not text:
            raise ValueError("empty model name")
        head, sep, rest = text.partition(":")
        if sep and head in KNOWN_PROVIDERS and rest:
            return cls(head, rest)
        return cls("openai", text)


@dataclass
class Completion:
    text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderError(RuntimeError):
    """A request failed in a way that retrying will not fix."""


def missing_keys(specs: list[ModelSpec]) -> list[str]:
    """Names of API-key variables the given models need but that are not set."""
    missing = set()
    for spec in specs:
        if spec.provider == "mock":
            continue
        key_var = PROVIDERS[spec.provider][2]
        if key_var and not provider_config(spec.provider).api_key:
            missing.add(key_var)
    return sorted(missing)


def list_models(provider: str) -> list[str]:
    config = provider_config(provider)
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    resp = httpx.get(f"{config.base_url}/models", headers=headers, timeout=30)
    if resp.status_code != 200:
        raise ProviderError(f"{provider}: HTTP {resp.status_code}: {resp.text[:300]}")
    return sorted(item["id"] for item in resp.json().get("data", []))


class OpenAICompatibleClient:
    def __init__(self, spec: ModelSpec, timeout: float = 180.0, max_retries: int = 4):
        self.spec = spec
        self.config = provider_config(spec.provider)
        self.timeout = timeout
        self.max_retries = max_retries

    async def complete(
        self,
        http: httpx.AsyncClient,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        context: Any = None,
    ) -> Completion:
        payload: dict[str, Any] = {"model": self.spec.model, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = f"{self.config.base_url}/chat/completions"

        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = await http.post(url, json=payload, headers=headers, timeout=self.timeout)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    return Completion(
                        text=(data["choices"][0]["message"].get("content") or "").strip(),
                        latency_ms=round((time.perf_counter() - started) * 1000),
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                    )
                if resp.status_code == 400 and "response_format" in payload and "response_format" in resp.text:
                    # This endpoint has no JSON mode; the prompt itself still asks for JSON.
                    payload.pop("response_format")
                    continue
                if resp.status_code not in RETRY_STATUS:
                    raise ProviderError(f"{self.spec.id}: HTTP {resp.status_code}: {resp.text[:300]}")
                last_error = f"HTTP {resp.status_code}"
            if attempt < self.max_retries:
                await asyncio.sleep(min(2**attempt, 30) + random.random())
        raise ProviderError(f"{self.spec.id}: gave up after {self.max_retries + 1} attempts ({last_error})")


# ---- mock ------------------------------------------------------------------------------------

MOCK_TIERS = {"strong": 0.85, "medium": 0.6, "weak": 0.35}
_QUALITY = re.compile(r"mock[ _]quality\"?\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _quality(text: str) -> float:
    match = _QUALITY.search(text)
    return float(match.group(1)) if match else 0.5


class MockClient:
    """Deterministic stand-in for a model or a judge, so the pipeline runs without an API key.

    Mock answers carry a visible "mock quality" score that the mock judge reads. The mock judge
    also leans towards the first answer on close calls, so the position-bias check has
    something to find.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.base_quality = MOCK_TIERS.get(spec.model, 0.5)

    async def complete(
        self,
        http: httpx.AsyncClient | None,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        context: Any = None,
    ) -> Completion:
        user = messages[-1]["content"]
        rng = _rng(self.spec.id, *(m["content"] for m in messages))
        task = context.get("task") if isinstance(context, dict) else None
        if task == "pairwise":
            text = self._pairwise(user, rng)
        elif task == "rubric":
            text = self._rubric(user, rng)
        else:
            text = self._answer(context, rng)
        return Completion(text=text, latency_ms=rng.randint(150, 1200))

    def _answer(self, prompt: Any, rng: random.Random) -> str:
        quality = min(1.0, max(0.0, self.base_quality + rng.uniform(-0.15, 0.15)))
        checks = {c["type"]: c.get("value") for c in getattr(prompt, "checks", ())}
        follows = rng.random() < quality
        if follows and checks.get("json_object"):
            return json.dumps({"answer": "mock answer", "mock_quality": round(quality, 2)})
        first = f"(mock quality: {quality:.2f}) Mock answer."
        if not follows:
            return first + " It ignores the formatting instructions."
        extra = []
        if "contains_all" in checks:
            extra.append(" ".join(checks["contains_all"]))
        if "contains_any" in checks:
            extra.append(checks["contains_any"][0])
        if "line_count" in checks:
            lines = [" ".join([first, *extra])]
            lines += [f"Line {i} of the mock answer." for i in range(2, checks["line_count"] + 1)]
        else:
            lines = [first, *extra]
            lines += [f"- Point {i}" for i in range(1, (checks.get("bullet_count") or 0) + 1)]
        text = "\n".join(lines)
        if checks.get("all_lowercase"):
            text = text.lower()
        if "ends_with" in checks:
            text = f"{text} {checks['ends_with']}"
        return text

    def _pairwise(self, user: str, rng: random.Random) -> str:
        before_b, _, answer_b = user.partition("[Answer B]")
        diff = _quality(before_b.partition("[Answer A]")[2]) - _quality(answer_b)
        if abs(diff) < 0.12:
            winner = "A" if rng.random() < 0.7 else "tie"
        else:
            better, worse = ("A", "B") if diff > 0 else ("B", "A")
            roll = rng.random()
            winner = better if roll < 0.85 else "tie" if roll < 0.95 else worse
        return json.dumps({"winner": winner, "rationale": "Mock verdict from the hidden quality scores."})

    def _rubric(self, user: str, rng: random.Random) -> str:
        from .judge import CRITERIA

        quality = _quality(user.partition("[Answer]")[2])
        scores = {c: max(1, min(5, round(1 + 4 * quality + rng.uniform(-0.7, 0.7)))) for c in CRITERIA}
        return json.dumps({**scores, "rationale": "Mock scores from the hidden quality score."})


def make_client(spec: ModelSpec) -> OpenAICompatibleClient | MockClient:
    return MockClient(spec) if spec.provider == "mock" else OpenAICompatibleClient(spec)
