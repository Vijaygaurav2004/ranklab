"""End-to-end runs against the mock provider: generate, judge, vote, rank and export."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from ranklab.generate import generate_answers
from ranklab.judge import combine_pairwise, judge_pairwise, judge_rubric, parse_pairwise, parse_rubric
from ranklab.prompts import default_prompts_path, load_prompts
from ranklab.providers import ModelSpec
from ranklab.report import markdown_report, preference_pairs, summarize
from ranklab.server import create_app
from ranklab.store import RunStore

MODELS = ["mock:strong", "mock:medium", "mock:weak"]
quiet = lambda _msg: None  # noqa: E731


@pytest.fixture
def store(tmp_path):
    store = RunStore(tmp_path, "test")
    specs = [ModelSpec.parse(m) for m in MODELS]
    store.create_or_update([s.id for s in specs], load_prompts(default_prompts_path()))
    stats = asyncio.run(generate_answers(store, specs, log=quiet))
    assert stats == {"done": 90, "failed": 0, "skipped": 0}
    return store


def test_model_spec_parsing():
    assert ModelSpec.parse("gpt-4o-mini").id == "openai:gpt-4o-mini"
    assert ModelSpec.parse("ollama:llama3.2:3b") == ModelSpec("ollama", "llama3.2:3b")
    assert ModelSpec.parse("openrouter:meta-llama/llama-3.1-8b-instruct").provider == "openrouter"


def test_generate_resumes_without_repeating_work(store):
    specs = [ModelSpec.parse(m) for m in MODELS]
    assert asyncio.run(generate_answers(store, specs, log=quiet)) == {"done": 0, "failed": 0, "skipped": 90}


def test_judge_ranks_mock_models_and_finds_position_bias(store):
    judge = ModelSpec.parse("mock:judge")
    store.set_judge(judge.id)
    assert asyncio.run(judge_rubric(store, judge, log=quiet))["done"] == 90
    assert asyncio.run(judge_pairwise(store, judge, log=quiet))["done"] == 180  # 90 pairs x both orders

    s = summarize(store, bootstrap_rounds=50)
    assert s["judge"]["pairs"] == 90
    assert [r["model"] for r in s["judge"]["leaderboard"]] == MODELS
    assert [r["model"] for r in s["rubric"]] == MODELS
    assert s["judge"]["inconsistent"] > 0
    assert s["judge"]["first_position_bias"] > s["judge"]["second_position_bias"]
    assert "Position-bias check" in markdown_report(s)


def test_combine_pairwise_flags_order_flips():
    records = [
        {"prompt_id": "p", "model_a": "x", "model_b": "y", "winner_pos": "A", "judge": "j"},
        {"prompt_id": "p", "model_a": "y", "model_b": "x", "winner_pos": "A", "judge": "j"},
        {"prompt_id": "q", "model_a": "x", "model_b": "y", "winner_pos": "B", "judge": "j"},
        {"prompt_id": "q", "model_a": "y", "model_b": "x", "winner_pos": "A", "judge": "j"},
    ]
    verdicts = {v["prompt_id"]: v for v in combine_pairwise(records)}
    assert verdicts["p"]["consistent"] is False
    assert verdicts["p"]["outcome"] == "tie"
    assert verdicts["p"]["position_bias"] == "first"
    assert verdicts["q"]["consistent"] is True
    assert verdicts["q"]["outcome"] == "y"


def test_judge_output_parsing():
    assert parse_pairwise('```json\n{"winner": "b", "rationale": "ok"}\n```')["winner_pos"] == "B"
    assert parse_pairwise('Verdict: {"winner": "tie"}')["winner_pos"] == "tie"
    with pytest.raises(ValueError):
        parse_pairwise('{"winner": "C"}')
    scores = parse_rubric('{"correctness": 5, "instruction_following": "4", "helpfulness": 3, "clarity": 2}')["scores"]
    assert scores == {"correctness": 5, "instruction_following": 4, "helpfulness": 3, "clarity": 2}
    with pytest.raises(ValueError):
        parse_rubric('{"correctness": 9, "instruction_following": 4, "helpfulness": 3, "clarity": 2}')


def test_voting_app_flow(store):
    client = TestClient(create_app(store, bootstrap_rounds=20))
    assert client.get("/").status_code == 200

    first = client.get("/api/next").json()
    assert first["done"] is False and first["left"] and first["right"]
    assert first["progress"] == {"voted": 0, "total": 90}
    assert "mock:" not in json.dumps(first["prompt"])  # nothing in the payload names a model

    resp = client.post("/api/vote", json={"ticket": first["ticket"], "choice": "left", "reason": "more accurate"})
    assert resp.status_code == 200
    assert resp.json()["progress"]["voted"] == 1
    assert client.post("/api/vote", json={"ticket": first["ticket"], "choice": "right"}).status_code == 409
    assert client.post("/api/vote", json={"ticket": "garbage", "choice": "left"}).status_code == 400

    board = client.get("/api/leaderboard").json()
    assert board["human"]["votes"] == 1
    assert len(board["human"]["leaderboard"]) == 3

    lines = client.get("/api/export.jsonl").text.strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["reason"] == "more accurate" and row["chosen"] and row["rejected"]
    assert row == preference_pairs(store, "human")[0]

    assert client.post("/api/undo").status_code == 200
    assert client.get("/api/run").json()["voted"] == 0
    assert client.post("/api/undo").status_code == 404
