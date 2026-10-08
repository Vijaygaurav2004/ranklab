"""Run storage: one folder per run with append-only JSONL files, so interrupted runs resume."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .prompts import Prompt, load_prompts, save_prompts

KINDS = ("answers", "rubric", "pairwise", "votes")


class RunStore:
    def __init__(self, runs_dir: Path | str, name: str):
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"bad run name {name!r}")
        self.name = name
        self.dir = Path(runs_dir) / name
        self._lock = threading.Lock()

    @property
    def config_path(self) -> Path:
        return self.dir / "run.json"

    @property
    def prompts_path(self) -> Path:
        return self.dir / "prompts.jsonl"

    def path(self, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"unknown record kind {kind!r}")
        return self.dir / f"{kind}.jsonl"

    def exists(self) -> bool:
        return self.config_path.is_file()

    def create_or_update(self, models: list[str], prompts: list[Prompt]) -> list[str]:
        """Add models to the run (creating it if needed). A run keeps the prompt set it started with."""
        self.dir.mkdir(parents=True, exist_ok=True)
        config = self.read_config() if self.exists() else {"name": self.name, "models": []}
        config["models"] += [m for m in models if m not in config["models"]]
        self._write_config(config)
        if not self.prompts_path.exists():
            save_prompts(prompts, self.prompts_path)
        return config["models"]

    def set_judge(self, judge: str) -> None:
        config = self.read_config()
        config["judge"] = judge
        self._write_config(config)

    def read_config(self) -> dict[str, Any]:
        if not self.exists():
            raise FileNotFoundError(f"run {self.name!r} not found in {self.dir.parent}; start it with `ranklab generate`")
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _write_config(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def models(self) -> list[str]:
        return list(self.read_config()["models"])

    def prompts(self) -> list[Prompt]:
        return load_prompts(self.prompts_path)

    def append(self, kind: str, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path(kind).open("a", encoding="utf-8") as fh:
                fh.write(line)

    def read(self, kind: str) -> list[dict[str, Any]]:
        path = self.path(kind)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def pop_last(self, kind: str) -> dict[str, Any] | None:
        with self._lock:
            rows = self.read(kind)
            if not rows:
                return None
            last = rows.pop()
            self.path(kind).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
            )
            return last


def latest_answers(store: RunStore) -> dict[tuple[str, str], dict[str, Any]]:
    """(model, prompt_id) -> the newest successful answer."""
    answers = {}
    for row in store.read("answers"):
        if not row.get("error"):
            answers[(row["model"], row["prompt_id"])] = row
    return answers
