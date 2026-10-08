"""Prompt sets (JSONL) and programmatic checks for instructions that can be verified without a model."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    text: str
    checks: tuple[dict[str, Any], ...] = ()
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "category": self.category, "prompt": self.text}
        if self.reference:
            data["reference"] = self.reference
        if self.checks:
            data["checks"] = list(self.checks)
        return data


def _words(text: str) -> int:
    return len(WORD.findall(text))


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text.strip()), dict)
    except json.JSONDecodeError:
        return False


def _has(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


CHECKS: dict[str, Callable[[str, Any], bool]] = {
    "max_words": lambda text, n: _words(text) <= n,
    "min_words": lambda text, n: _words(text) >= n,
    "bullet_count": lambda text, n: sum(1 for line in _lines(text) if BULLET.match(line)) == n,
    "line_count": lambda text, n: len(_lines(text)) == n,
    "json_object": lambda text, want: _is_json_object(text) == bool(want),
    "contains_all": lambda text, needles: all(_has(text, s) for s in needles),
    "contains_any": lambda text, needles: any(_has(text, s) for s in needles),
    "contains_none": lambda text, needles: not any(_has(text, s) for s in needles),
    "ends_with": lambda text, suffix: text.rstrip().endswith(suffix),
    "all_lowercase": lambda text, want: (text == text.lower()) == bool(want),
    "regex": lambda text, pattern: re.search(pattern, text, re.IGNORECASE | re.MULTILINE) is not None,
}


def run_checks(answer: str, checks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": c["type"], "value": c.get("value"), "passed": bool(CHECKS[c["type"]](answer, c.get("value")))}
        for c in checks
    ]


def default_prompts_path() -> Path:
    return Path(__file__).with_name("data") / "prompts.jsonl"


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from exc
        pid, text = row.get("id"), row.get("prompt")
        if not pid or not text:
            raise ValueError(f"{path}:{n}: each line needs an 'id' and a 'prompt'")
        if pid in seen:
            raise ValueError(f"{path}:{n}: duplicate id {pid!r}")
        seen.add(pid)
        checks = tuple(row.get("checks", ()))
        for check in checks:
            if check.get("type") not in CHECKS:
                raise ValueError(f"{path}:{n}: unknown check type {check.get('type')!r}")
            if check["type"] == "regex":
                re.compile(check["value"])
        prompts.append(Prompt(pid, row.get("category", "general"), text, checks, row.get("reference")))
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def save_prompts(prompts: Iterable[Prompt], path: Path) -> None:
    path.write_text("".join(json.dumps(p.to_dict(), ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8")
