"""AI judge: rubric scores for each answer, and pairwise verdicts asked in both answer orders.

Asking every pairwise question twice, with the answers swapped, exposes position bias: a fair
judge picks the same winner either way. Verdicts that flip are recorded but scored as ties.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from itertools import combinations
from typing import Any

import httpx

from .prompts import Prompt
from .providers import ModelSpec, ProviderError, make_client
from .store import RunStore, latest_answers

CRITERIA = ("correctness", "instruction_following", "helpfulness", "clarity")

RUBRIC_SYSTEM = """You are an impartial evaluator of AI assistant answers.
Score the answer from 1 (very poor) to 5 (excellent) on each criterion:
- correctness: facts, logic and code are right. If the prompt rests on a false premise, correcting it is correct; going along with it is not.
- instruction_following: obeys every explicit constraint (format, length, style, words to use or avoid).
- helpfulness: actually solves the user's problem at the right level of detail.
- clarity: well organised and easy to read.
Judge substance, not confidence or length. If a reference answer is given, use it to check correctness.
Reply with only a JSON object: {"correctness": 1-5, "instruction_following": 1-5, "helpfulness": 1-5, "clarity": 1-5, "rationale": "one or two sentences"}"""

PAIRWISE_SYSTEM = """You are an impartial judge comparing two AI assistant answers to the same prompt.
Pick the better answer overall: correct, follows every instruction, helpful and clear.
- The order of the answers is random. Do not favour an answer because it comes first or second.
- Do not favour the longer answer. A short correct answer beats a long wrong one.
- If the prompt rests on a false premise, the answer that corrects it is better.
- If a reference answer is given, use it to check correctness.
Say "tie" only when the answers are genuinely equal in quality.
Reply with only a JSON object: {"winner": "A" or "B" or "tie", "rationale": "one or two sentences"}"""

Job = tuple[dict[str, Any], list[dict[str, str]], Callable[[str], dict[str, Any]], str]


def _reference(prompt: Prompt) -> str:
    return f"\n\n[Reference answer]\n{prompt.reference}" if prompt.reference else ""


def rubric_messages(prompt: Prompt, answer: str) -> list[dict[str, str]]:
    user = f"[Prompt]\n{prompt.text}{_reference(prompt)}\n\n[Answer]\n{answer}"
    return [{"role": "system", "content": RUBRIC_SYSTEM}, {"role": "user", "content": user}]


def pairwise_messages(prompt: Prompt, answer_a: str, answer_b: str) -> list[dict[str, str]]:
    user = f"[Prompt]\n{prompt.text}{_reference(prompt)}\n\n[Answer A]\n{answer_a}\n\n[Answer B]\n{answer_b}"
    return [{"role": "system", "content": PAIRWISE_SYSTEM}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"judge did not return JSON: {text[:120]!r}") from None
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("judge returned JSON that is not an object")
    return obj


def parse_rubric(text: str) -> dict[str, Any]:
    obj = parse_json_object(text)
    scores = {}
    for criterion in CRITERIA:
        value = obj.get(criterion)
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError(f"bad or missing score for {criterion}: {value!r}")
        scores[criterion] = round(value)
    return {"scores": scores, "rationale": str(obj.get("rationale", ""))[:1000]}


def parse_pairwise(text: str) -> dict[str, Any]:
    obj = parse_json_object(text)
    winner = str(obj.get("winner", "")).strip().upper()
    if winner not in ("A", "B", "TIE"):
        raise ValueError(f"bad winner {obj.get('winner')!r}")
    return {"winner_pos": "tie" if winner == "TIE" else winner, "rationale": str(obj.get("rationale", ""))[:1000]}


async def judge_rubric(
    store: RunStore, judge: ModelSpec, concurrency: int = 4, log: Callable[[str], None] = print
) -> dict[str, int]:
    prompts = {p.id: p for p in store.prompts()}
    answers = latest_answers(store)
    done = {
        (r["model"], r["prompt_id"]) for r in store.read("rubric") if not r.get("error") and r.get("judge") == judge.id
    }
    jobs: list[Job] = [
        ({"model": model, "prompt_id": pid}, rubric_messages(prompts[pid], row["answer"]), parse_rubric, "rubric")
        for (model, pid), row in answers.items()
        if pid in prompts and (model, pid) not in done
    ]
    return await _run(store, "rubric", judge, jobs, concurrency, log)


async def judge_pairwise(
    store: RunStore, judge: ModelSpec, concurrency: int = 4, log: Callable[[str], None] = print
) -> dict[str, int]:
    answers = latest_answers(store)
    models = store.models()
    done = {
        (r["prompt_id"], r["model_a"], r["model_b"])
        for r in store.read("pairwise")
        if not r.get("error") and r.get("judge") == judge.id
    }
    jobs: list[Job] = []
    for prompt in store.prompts():
        ready = sorted(m for m in models if (m, prompt.id) in answers)
        for m1, m2 in combinations(ready, 2):
            for a, b in ((m1, m2), (m2, m1)):
                if (prompt.id, a, b) in done:
                    continue
                messages = pairwise_messages(prompt, answers[(a, prompt.id)]["answer"], answers[(b, prompt.id)]["answer"])
                jobs.append(({"prompt_id": prompt.id, "model_a": a, "model_b": b}, messages, parse_pairwise, "pairwise"))
    return await _run(store, "pairwise", judge, jobs, concurrency, log)


async def _run(
    store: RunStore, kind: str, judge: ModelSpec, jobs: list[Job], concurrency: int, log: Callable[[str], None]
) -> dict[str, int]:
    stats = {"done": 0, "failed": 0}
    if not jobs:
        return stats
    client = make_client(judge)
    sem = asyncio.Semaphore(concurrency)

    async def one(http: httpx.AsyncClient, ids: dict[str, Any], messages, parse, task: str) -> None:
        async with sem:
            record = {**ids, "judge": judge.id, "ts": time.time()}
            try:
                result = await client.complete(http, messages, json_mode=True, context={"task": task})
                record.update(parse(result.text))
                stats["done"] += 1
            except (ProviderError, ValueError) as exc:
                record["error"] = str(exc)
                stats["failed"] += 1
                log(f"  ! {kind} {ids}: {exc}")
            store.append(kind, record)
            finished = stats["done"] + stats["failed"]
            if finished % 20 == 0 or finished == len(jobs):
                log(f"  {finished}/{len(jobs)} {kind} judgements")

    async with httpx.AsyncClient() as http:
        await asyncio.gather(*(one(http, *job) for job in jobs))
    return stats


def _winner(record: dict[str, Any]) -> str:
    return {"A": record["model_a"], "B": record["model_b"]}.get(record["winner_pos"], "tie")


def combine_pairwise(records: list[dict[str, Any]], judge: str | None = None) -> list[dict[str, Any]]:
    """Merge the two orderings of each judged pair into one verdict.

    The verdict stands only if both orderings name the same winner (or both say tie). Otherwise
    it is scored as a tie and flagged inconsistent; when the judge picked the same *position*
    both times, that is recorded as first- or second-position bias.
    """
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in records:
        if not r.get("error") and (judge is None or r.get("judge") == judge):
            latest[(r["prompt_id"], r["model_a"], r["model_b"])] = r
    verdicts = []
    for (pid, a, b), first in latest.items():
        if a > b:
            continue  # visit each unordered pair once, from its sorted ordering
        second = latest.get((pid, b, a))
        if second is None:
            continue
        w1, w2 = _winner(first), _winner(second)
        consistent = w1 == w2
        bias = None
        if not consistent and first["winner_pos"] == second["winner_pos"] != "tie":
            bias = "first" if first["winner_pos"] == "A" else "second"
        verdicts.append(
            {
                "prompt_id": pid,
                "models": [a, b],
                "outcome": w1 if consistent else "tie",
                "consistent": consistent,
                "position_bias": bias,
                "judge": first.get("judge"),
            }
        )
    return verdicts
