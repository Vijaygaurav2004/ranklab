"""Ratings and agreement statistics, in plain Python.

Ratings use the Bradley-Terry model (the pairwise model behind arena-style LLM leaderboards),
fitted with Hunter's MM algorithm and reported on an Elo-like scale where 1000 is average.
Confidence intervals come from bootstrap resampling of the battles.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

Battle = tuple[str, str, str]  # (model_a, model_b, outcome); outcome is model_a, model_b or "tie"


def bradley_terry(
    battles: Sequence[Battle], models: Sequence[str], prior: float = 1.0, iters: int = 1000, tol: float = 1e-10
) -> dict[str, float]:
    """Strength per model (geometric mean 1). Ties count as half a win for each side.

    `prior` adds that many virtual tied games between every pair of models, which keeps
    estimates finite when a model has won or lost everything (a mild pull towards average).
    """
    n = len(models)
    if n == 0:
        return {}
    index = {m: i for i, m in enumerate(models)}
    wins = [0.0] * n
    games = [[0.0] * n for _ in range(n)]
    for a, b, outcome in battles:
        i, j = index[a], index[b]
        games[i][j] += 1
        games[j][i] += 1
        if outcome == a:
            wins[i] += 1
        elif outcome == b:
            wins[j] += 1
        else:
            wins[i] += 0.5
            wins[j] += 0.5
    for i in range(n):
        for j in range(i + 1, n):
            games[i][j] += prior
            games[j][i] += prior
            wins[i] += prior / 2
            wins[j] += prior / 2

    strength = [1.0] * n
    for _ in range(iters):
        updated = []
        for i in range(n):
            denom = sum(games[i][j] / (strength[i] + strength[j]) for j in range(n) if j != i and games[i][j])
            updated.append(wins[i] / denom if denom and wins[i] > 0 else strength[i])
        log_mean = sum(math.log(s) for s in updated) / n
        updated = [s / math.exp(log_mean) for s in updated]
        delta = max(abs(x - y) for x, y in zip(updated, strength))
        strength = updated
        if delta < tol:
            break
    return {m: strength[index[m]] for m in models}


def elo_scale(strengths: dict[str, float]) -> dict[str, float]:
    return {m: 1000 + 400 * math.log10(s) for m, s in strengths.items()}


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of an empty list")
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def ratings_with_ci(
    battles: Sequence[Battle], models: Sequence[str], rounds: int = 300, seed: int = 7, prior: float = 1.0
) -> dict[str, dict[str, float]]:
    point = elo_scale(bradley_terry(battles, models, prior))
    samples: dict[str, list[float]] = {m: [] for m in models}
    rng = random.Random(seed)
    for _ in range(rounds if battles else 0):
        resample = [battles[rng.randrange(len(battles))] for _ in range(len(battles))]
        for m, rating in elo_scale(bradley_terry(resample, models, prior)).items():
            samples[m].append(rating)
    out = {}
    for m in models:
        values = sorted(samples[m]) or [point[m]]
        out[m] = {"rating": point[m], "ci_low": percentile(values, 0.025), "ci_high": percentile(values, 0.975)}
    return out


def leaderboard(battles: Sequence[Battle], models: Sequence[str], rounds: int = 300) -> list[dict[str, Any]]:
    record = {m: {"wins": 0, "losses": 0, "ties": 0} for m in models}
    for a, b, outcome in battles:
        if outcome == "tie":
            record[a]["ties"] += 1
            record[b]["ties"] += 1
        else:
            record[outcome]["wins"] += 1
            record[b if outcome == a else a]["losses"] += 1
    ratings = ratings_with_ci(battles, models, rounds)
    rows = []
    for m in models:
        r = record[m]
        played = r["wins"] + r["losses"] + r["ties"]
        rows.append(
            {
                "model": m,
                **ratings[m],
                **r,
                "battles": played,
                "win_rate": (r["wins"] + 0.5 * r["ties"]) / played if played else None,
            }
        )
    rows.sort(key=lambda row: row["rating"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Agreement between two raters, corrected for the agreement expected by chance."""
    n = len(pairs)
    if n == 0:
        return None
    labels = {label for pair in pairs for label in pair}
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(
        (sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n) for label in labels
    )
    if expected >= 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def relative_label(models: Sequence[str], outcome: str) -> str:
    """An outcome relative to the sorted pair: 'first', 'second' or 'tie'."""
    first, second = sorted(models)
    return "first" if outcome == first else "second" if outcome == second else "tie"


def agreement(judge: dict[tuple[str, str, str], str], human: dict[tuple[str, str, str], str]) -> dict[str, Any]:
    """Compare judge and human verdicts keyed by (prompt_id, model, model) with models sorted."""
    common = sorted(set(judge) & set(human))
    labelled = [(relative_label(k[1:], judge[k]), relative_label(k[1:], human[k])) for k in common]
    decisive = [(j, h) for j, h in labelled if j != "tie" and h != "tie"]
    return {
        "pairs": len(common),
        "decisive_pairs": len(decisive),
        "agreement_rate": sum(j == h for j, h in decisive) / len(decisive) if decisive else None,
        "kappa": cohens_kappa(labelled),
    }
