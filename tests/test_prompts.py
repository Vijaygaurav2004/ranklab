import json

import pytest

from ranklab.prompts import default_prompts_path, load_prompts, run_checks


def passed(answer, *checks):
    return [r["passed"] for r in run_checks(answer, checks)]


def test_builtin_prompt_set_loads():
    prompts = load_prompts(default_prompts_path())
    assert len(prompts) == 30
    assert {p.category for p in prompts} == {"reasoning", "coding", "writing", "factual", "instructions", "false-premise"}


def test_word_and_line_checks():
    assert passed("one two three", {"type": "max_words", "value": 3}, {"type": "min_words", "value": 4}) == [True, False]
    assert passed("a\n\nb\nc", {"type": "line_count", "value": 3}) == [True]
    assert passed("Intro\n- one\n* two\n3. three", {"type": "bullet_count", "value": 3}) == [True]


def test_json_lowercase_and_ending_checks():
    assert passed('{"benefits": ["x"]}', {"type": "json_object", "value": True}) == [True]
    assert passed('```json\n{"a": 1}\n```', {"type": "json_object", "value": True}) == [False]
    assert passed("all quiet here", {"type": "all_lowercase", "value": True}) == [True]
    assert passed("Not quiet", {"type": "all_lowercase", "value": True}) == [False]
    assert passed("Done. Any questions?  ", {"type": "ends_with", "value": "Any questions?"}) == [True]


def test_contains_and_regex_checks():
    text = "Saturn has the most moons. Answer: 5 minutes"
    assert passed(
        text,
        {"type": "contains_all", "value": ["saturn", "moons"]},
        {"type": "contains_any", "value": ["jupiter", "MOONS"]},
        {"type": "contains_none", "value": ["pluto"]},
        {"type": "regex", "value": r"answer:\s*5\b"},
        {"type": "regex", "value": r"answer:\s*50\b"},
    ) == [True, True, True, True, False]


def test_load_rejects_unknown_check(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": "x", "prompt": "hi", "checks": [{"type": "vibes"}]}) + "\n")
    with pytest.raises(ValueError, match="unknown check type"):
        load_prompts(path)
