import pytest

from ranklab.metrics import agreement, bradley_terry, cohens_kappa, elo_scale, leaderboard, percentile


def test_bradley_terry_matches_win_share_without_prior():
    battles = [("a", "b", "a")] * 8 + [("a", "b", "b")] * 2
    s = bradley_terry(battles, ["a", "b"], prior=0)
    assert s["a"] / (s["a"] + s["b"]) == pytest.approx(0.8, abs=1e-6)


def test_ties_only_means_equal_ratings_at_1000():
    ratings = elo_scale(bradley_terry([("a", "b", "tie")] * 5, ["a", "b"]))
    assert ratings["a"] == pytest.approx(1000)
    assert ratings["b"] == pytest.approx(1000)


def test_prior_keeps_an_unbeaten_model_finite():
    ratings = elo_scale(bradley_terry([("a", "b", "a")] * 5, ["a", "b"], prior=1.0))
    assert 1000 < ratings["a"] < 1400
    assert ratings["a"] + ratings["b"] == pytest.approx(2000)


def test_leaderboard_orders_models_and_brackets_ratings():
    battles = (
        [("a", "b", "a")] * 7 + [("a", "b", "b")] * 3
        + [("b", "c", "b")] * 7 + [("b", "c", "c")] * 3
        + [("a", "c", "a")] * 9 + [("a", "c", "c")] * 1
    )
    rows = leaderboard(battles, ["c", "a", "b"], rounds=200)
    assert [r["model"] for r in rows] == ["a", "b", "c"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    for row in rows:
        assert row["ci_low"] <= row["rating"] <= row["ci_high"]
    top = rows[0]
    assert (top["wins"], top["losses"], top["ties"], top["battles"]) == (16, 4, 0, 20)
    assert top["win_rate"] == pytest.approx(0.8)


def test_percentile_interpolates():
    assert percentile([0, 10], 0.25) == pytest.approx(2.5)
    assert percentile([5], 0.975) == 5


def test_cohens_kappa_textbook_example():
    pairs = [("x", "x")] * 20 + [("x", "y")] * 5 + [("y", "x")] * 10 + [("y", "y")] * 15
    assert cohens_kappa(pairs) == pytest.approx(0.4)
    assert cohens_kappa([("x", "x"), ("y", "y")]) == pytest.approx(1.0)
    assert cohens_kappa([]) is None


def test_agreement_ignores_ties_for_the_rate():
    judge = {("p1", "a", "b"): "a", ("p2", "a", "b"): "tie", ("p3", "a", "b"): "b"}
    human = {("p1", "a", "b"): "a", ("p2", "a", "b"): "a", ("p3", "a", "b"): "a", ("p4", "a", "b"): "b"}
    result = agreement(judge, human)
    assert result["pairs"] == 3
    assert result["decisive_pairs"] == 2
    assert result["agreement_rate"] == pytest.approx(0.5)
