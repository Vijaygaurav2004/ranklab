"""Everything the leaderboard shows, as one JSON-able summary, plus the Markdown report and preference export."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from .judge import CRITERIA, combine_pairwise
from .metrics import agreement, leaderboard
from .prompts import Prompt, run_checks
from .store import RunStore, latest_answers

PairKey = tuple[str, str, str]  # (prompt_id, model, model) with the two models sorted
FEW_BATTLES = 10  # below this many comparisons per model, rankings are flagged as rough


def pair_key(prompt_id: str, m1: str, m2: str) -> PairKey:
    a, b = sorted((m1, m2))
    return (prompt_id, a, b)


def vote_outcome(vote: dict[str, Any]) -> str:
    """'left'/'right' name a winner; 'tie' and 'both_bad' both count as a tie."""
    return {"left": vote["left"], "right": vote["right"]}.get(vote["choice"], "tie")


def human_votes(store: RunStore) -> dict[PairKey, dict[str, Any]]:
    """The latest human vote for each (prompt, model pair)."""
    return {pair_key(v["prompt_id"], v["left"], v["right"]): v for v in store.read("votes")}


def votable_pairs(store: RunStore) -> list[PairKey]:
    """Every (prompt, model pair) where both models have an answer."""
    answers = latest_answers(store)
    models = store.models()
    pairs = []
    for prompt in store.prompts():
        ready = sorted(m for m in models if (m, prompt.id) in answers)
        pairs += [(prompt.id, a, b) for i, a in enumerate(ready) for b in ready[i + 1 :]]
    return pairs


def summarize(store: RunStore, bootstrap_rounds: int = 300) -> dict[str, Any]:
    config = store.read_config()
    models: list[str] = config["models"]
    judge_id: str | None = config.get("judge")
    prompts = {p.id: p for p in store.prompts()}
    answers = latest_answers(store)
    failed = {(r["model"], r["prompt_id"]) for r in store.read("answers") if r.get("error")} - set(answers)

    votes = human_votes(store)
    human_battles = [
        (v["left"], v["right"], vote_outcome(v)) for v in votes.values() if {v["left"], v["right"]} <= set(models)
    ]

    verdicts = [v for v in combine_pairwise(store.read("pairwise"), judge_id) if set(v["models"]) <= set(models)]
    judge_battles = [(v["models"][0], v["models"][1], v["outcome"]) for v in verdicts]
    inconsistent = [v for v in verdicts if not v["consistent"]]

    return {
        "run": store.name,
        "models": models,
        "prompts": len(prompts),
        "categories": sorted({p.category for p in prompts.values()}),
        "answers": len(answers),
        "failed_answers": len(failed),
        "human": {
            "votes": len(votes),
            "pairs_total": len(votable_pairs(store)),
            "leaderboard": leaderboard(human_battles, models, bootstrap_rounds) if human_battles else [],
        },
        "judge": {
            "model": judge_id,
            "pairs": len(verdicts),
            "leaderboard": leaderboard(judge_battles, models, bootstrap_rounds) if judge_battles else [],
            "inconsistent": len(inconsistent),
            "inconsistency_rate": len(inconsistent) / len(verdicts) if verdicts else None,
            "first_position_bias": sum(v["position_bias"] == "first" for v in inconsistent),
            "second_position_bias": sum(v["position_bias"] == "second" for v in inconsistent),
        },
        "agreement": agreement(
            {(v["prompt_id"], *v["models"]): v["outcome"] for v in verdicts},
            {key: vote_outcome(v) for key, v in votes.items()},
        ),
        **_rubric_summary(store, models, prompts, judge_id),
        "checks": _check_summary(models, prompts, answers),
    }


def _rubric_summary(
    store: RunStore, models: list[str], prompts: dict[str, Prompt], judge_id: str | None
) -> dict[str, Any]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in store.read("rubric"):
        if not r.get("error") and (judge_id is None or r.get("judge") == judge_id):
            latest[(r["model"], r["prompt_id"])] = r
    rubric, by_category = [], {}
    for model in models:
        rows = [r for (m, pid), r in latest.items() if m == model and pid in prompts]
        if not rows:
            continue
        entry: dict[str, Any] = {"model": model, "n": len(rows)}
        for criterion in CRITERIA:
            entry[criterion] = mean(r["scores"][criterion] for r in rows)
        entry["overall"] = mean(entry[c] for c in CRITERIA)
        rubric.append(entry)
        per_category: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            per_category[prompts[r["prompt_id"]].category].append(mean(r["scores"][c] for c in CRITERIA))
        by_category[model] = {category: mean(values) for category, values in per_category.items()}
    rubric.sort(key=lambda e: e["overall"], reverse=True)
    return {"rubric": rubric, "by_category": by_category}


def _check_summary(
    models: list[str], prompts: dict[str, Prompt], answers: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        results = [
            all(c["passed"] for c in run_checks(row["answer"], prompts[pid].checks))
            for (m, pid), row in answers.items()
            if m == model and pid in prompts and prompts[pid].checks
        ]
        if results:
            rows.append({"model": model, "passed": sum(results), "total": len(results), "rate": sum(results) / len(results)})
    rows.sort(key=lambda r: r["rate"], reverse=True)
    return rows


def preference_pairs(store: RunStore, source: str = "human") -> list[dict[str, Any]]:
    """Chosen/rejected pairs in the shape preference-tuning (RLHF reward models, DPO) expects.
    Ties and inconsistent judge verdicts are left out: they carry no preference."""
    if source not in ("human", "judge", "all"):
        raise ValueError(f"unknown source {source!r}")
    prompts = {p.id: p for p in store.prompts()}
    answers = latest_answers(store)
    picks: list[tuple[str, str, str, str, str | None]] = []  # (prompt_id, chosen, rejected, source, reason)
    if source in ("human", "all"):
        for v in human_votes(store).values():
            if v["choice"] in ("left", "right"):
                chosen, rejected = (v["left"], v["right"]) if v["choice"] == "left" else (v["right"], v["left"])
                picks.append((v["prompt_id"], chosen, rejected, "human", v.get("reason") or None))
    if source in ("judge", "all"):
        for verdict in combine_pairwise(store.read("pairwise"), store.read_config().get("judge")):
            if verdict["consistent"] and verdict["outcome"] != "tie":
                a, b = verdict["models"]
                rejected = b if verdict["outcome"] == a else a
                picks.append((verdict["prompt_id"], verdict["outcome"], rejected, "judge", None))
    rows = []
    for pid, chosen, rejected, origin, reason in picks:
        if pid not in prompts or (chosen, pid) not in answers or (rejected, pid) not in answers:
            continue
        row = {
            "prompt_id": pid,
            "prompt": prompts[pid].text,
            "chosen": answers[(chosen, pid)]["answer"],
            "rejected": answers[(rejected, pid)]["answer"],
            "chosen_model": chosen,
            "rejected_model": rejected,
            "source": origin,
        }
        if reason:
            row["reason"] = reason
        rows.append(row)
    return rows


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def markdown_report(s: dict[str, Any]) -> str:
    lines = [
        f"# RankLab report: {s['run']}",
        "",
        f"{s['prompts']} prompts, {len(s['models'])} models, {s['answers']} answers"
        + (f" ({s['failed_answers']} failed)" if s["failed_answers"] else "")
        + ".",
        "",
    ]

    def ranking(title: str, rows: list[dict[str, Any]], empty: str) -> None:
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend([empty, ""])
            return
        lines.extend(["| Rank | Model | Rating | 95% CI | Win rate | W-L-T |", "|---|---|---:|---:|---:|---:|"])
        for r in rows:
            lines.append(
                f"| {r['rank']} | {r['model']} | {r['rating']:.0f} | {r['ci_low']:.0f} to {r['ci_high']:.0f} "
                f"| {_pct(r['win_rate'])} | {r['wins']}-{r['losses']}-{r['ties']} |"
            )
        lines.append("")
        if any(r["battles"] < FEW_BATTLES for r in rows):
            lines.extend([f"Some models have fewer than {FEW_BATTLES} comparisons; treat this ranking as rough.", ""])

    human, judge, agree = s["human"], s["judge"], s["agreement"]
    ranking("Human preference ranking", human["leaderboard"], "No human votes yet. Run `ranklab serve` and vote.")
    lines.extend([f"Human votes: {human['votes']} of {human['pairs_total']} pairs.", ""])
    ranking(f"AI judge ranking ({judge['model'] or 'no judge yet'})", judge["leaderboard"], "No judge verdicts yet. Run `ranklab judge`.")
    if judge["pairs"]:
        lines.extend(
            [
                f"Position-bias check: {judge['inconsistent']} of {judge['pairs']} pairs ({_pct(judge['inconsistency_rate'])}) "
                f"got a different verdict when the answer order was swapped; {judge['first_position_bias']} favoured the "
                f"first answer both times and {judge['second_position_bias']} the second.",
                "",
            ]
        )
    if agree["pairs"]:
        kappa = "n/a" if agree["kappa"] is None else f"{agree['kappa']:.2f}"
        lines.extend(
            [
                "## Judge vs human",
                "",
                f"{agree['pairs']} pairs rated by both. The judge agreed with the human on "
                f"{_pct(agree['agreement_rate'])} of {agree['decisive_pairs']} pairs where both picked a winner; "
                f"Cohen's kappa over all {agree['pairs']} pairs is {kappa}.",
                "",
            ]
        )
    if s["rubric"]:
        lines.extend(["## Rubric scores (1-5, AI judge)", "", "| Model | Correct | Follows instructions | Helpful | Clear | Overall |", "|---|---:|---:|---:|---:|---:|"])
        for r in s["rubric"]:
            lines.append(
                f"| {r['model']} | {r['correctness']:.2f} | {r['instruction_following']:.2f} | {r['helpfulness']:.2f} "
                f"| {r['clarity']:.2f} | {r['overall']:.2f} |"
            )
        lines.append("")
    if s["checks"]:
        lines.extend(["## Automatic instruction checks", "", "| Model | Passed | Pass rate |", "|---|---:|---:|"])
        for r in s["checks"]:
            lines.append(f"| {r['model']} | {r['passed']} / {r['total']} | {_pct(r['rate'])} |")
        lines.append("")
    return "\n".join(lines)
