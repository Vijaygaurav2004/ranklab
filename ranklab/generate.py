"""Collect one answer per (model, prompt), skipping pairs a previous run already answered."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import httpx

from .prompts import Prompt
from .providers import ModelSpec, ProviderError, make_client
from .store import RunStore, latest_answers


async def generate_answers(
    store: RunStore,
    specs: list[ModelSpec],
    concurrency: int = 4,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    prompts = store.prompts()
    have = latest_answers(store)
    todo = [(spec, p) for spec in specs for p in prompts if (spec.id, p.id) not in have]
    stats = {"done": 0, "failed": 0, "skipped": len(specs) * len(prompts) - len(todo)}
    if not todo:
        return stats

    clients = {spec.id: make_client(spec) for spec in specs}
    sem = asyncio.Semaphore(concurrency)

    async def one(http: httpx.AsyncClient, spec: ModelSpec, prompt: Prompt) -> None:
        async with sem:
            record = {"model": spec.id, "prompt_id": prompt.id, "ts": time.time()}
            try:
                result = await clients[spec.id].complete(
                    http, [{"role": "user", "content": prompt.text}], context=prompt
                )
            except ProviderError as exc:
                record["error"] = str(exc)
                stats["failed"] += 1
                log(f"  ! {spec.id} on {prompt.id}: {exc}")
            else:
                record.update(
                    answer=result.text,
                    latency_ms=result.latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
                stats["done"] += 1
            store.append("answers", record)
            finished = stats["done"] + stats["failed"]
            if finished % 10 == 0 or finished == len(todo):
                log(f"  {finished}/{len(todo)} answers")

    async with httpx.AsyncClient() as http:
        await asyncio.gather(*(one(http, spec, p) for spec, p in todo))
    return stats
