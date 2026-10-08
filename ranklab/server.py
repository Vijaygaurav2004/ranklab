"""Local web app: blind pairwise voting plus the live leaderboard."""

from __future__ import annotations

import base64
import binascii
import json
import random
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .report import human_votes, pair_key, preference_pairs, summarize, votable_pairs
from .store import RunStore, latest_answers

STATIC_DIR = Path(__file__).with_name("static")


class VoteIn(BaseModel):
    ticket: str
    choice: Literal["left", "right", "tie", "both_bad"]
    reason: str = Field(default="", max_length=1000)


def _encode(data: dict[str, str]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def _decode(ticket: str) -> dict[str, str]:
    try:
        data = json.loads(base64.urlsafe_b64decode(ticket.encode()))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise HTTPException(400, "bad ticket") from None
    if not isinstance(data, dict) or not {"p", "l", "r"} <= set(data):
        raise HTTPException(400, "bad ticket")
    return data


def create_app(store: RunStore, bootstrap_rounds: int = 300) -> FastAPI:
    store.read_config()  # fail fast if the run does not exist
    app = FastAPI(title="RankLab", docs_url=None, redoc_url=None, openapi_url=None)
    rng = random.Random()

    def progress() -> dict[str, int]:
        return {"voted": len(human_votes(store)), "total": len(votable_pairs(store))}

    @app.get("/api/run")
    def run_info() -> dict[str, Any]:
        config = store.read_config()
        return {"name": store.name, "models": config["models"], "judge": config.get("judge"), **progress()}

    @app.get("/api/next")
    def next_pair() -> dict[str, Any]:
        voted = set(human_votes(store))
        pending = [key for key in votable_pairs(store) if key not in voted]
        if not pending:
            return {"done": True, "progress": progress()}
        pid, a, b = rng.choice(pending)
        left, right = (a, b) if rng.random() < 0.5 else (b, a)  # random sides, so position cannot leak the model
        prompt = next(p for p in store.prompts() if p.id == pid)
        answers = latest_answers(store)
        return {
            "done": False,
            "ticket": _encode({"p": pid, "l": left, "r": right}),
            "prompt": {"id": pid, "category": prompt.category, "text": prompt.text},
            "left": answers[(left, pid)]["answer"],
            "right": answers[(right, pid)]["answer"],
            "progress": progress(),
        }

    @app.post("/api/vote")
    def vote(body: VoteIn) -> dict[str, Any]:
        ticket = _decode(body.ticket)
        key = pair_key(ticket["p"], ticket["l"], ticket["r"])
        if key not in set(votable_pairs(store)):
            raise HTTPException(400, "unknown pair")
        if key in human_votes(store):
            raise HTTPException(409, "you already voted on this pair")
        store.append(
            "votes",
            {
                "prompt_id": ticket["p"],
                "left": ticket["l"],
                "right": ticket["r"],
                "choice": body.choice,
                "reason": body.reason.strip(),
                "ts": time.time(),
            },
        )
        return {"left_model": ticket["l"], "right_model": ticket["r"], "progress": progress()}

    @app.post("/api/undo")
    def undo() -> dict[str, Any]:
        last = store.pop_last("votes")
        if last is None:
            raise HTTPException(404, "no votes to undo")
        return {"undone": last["prompt_id"], "progress": progress()}

    @app.get("/api/leaderboard")
    def board() -> dict[str, Any]:
        return summarize(store, bootstrap_rounds)

    @app.get("/api/export.jsonl")
    def export(source: Literal["human", "judge", "all"] = "human") -> PlainTextResponse:
        rows = preference_pairs(store, source)
        return PlainTextResponse(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{store.name}-preferences-{source}.jsonl"'},
        )

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
