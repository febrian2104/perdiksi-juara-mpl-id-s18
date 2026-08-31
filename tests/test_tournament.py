import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpl_predictor.models.season18_predictions import build_live_prediction_windows
from mpl_predictor.models.tournament import (
    adjusted_series_probability,
    simulate_playoff_bracket,
    validate_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation_config.json"


@pytest.fixture
def season18_format() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_season18_format_is_confirmed_and_declarative(season18_format: dict) -> None:
    summary = validate_simulation_config(season18_format)

    assert summary["season"] == 18
    assert summary["playoff_team_count"] == 6
    assert summary["bracket_match_count"] == 8
    assert summary["round_counts"] == {
        "play_ins": 2,
        "upper_semifinal": 2,
        "lower_semifinal": 1,
        "upper_final": 1,
        "lower_final": 1,
        "grand_final": 1,
    }


def test_unconfirmed_future_format_is_rejected(season18_format: dict) -> None:
    future = copy.deepcopy(season18_format)
    future["season"] = 19
    future["format_confirmation"]["season"] = 19
    future["format_confirmation"]["status"] = "unconfirmed"

    with pytest.raises(ValueError, match="format is not confirmed"):
        validate_simulation_config(future)


def test_forward_bracket_reference_is_rejected(season18_format: dict) -> None:
    invalid = copy.deepcopy(season18_format)
    invalid["playoffs"]["bracket"][0]["team_a"] = {
        "source": "winner",
        "match_id": "grand_final",
    }

    with pytest.raises(ValueError, match="unavailable match"):
        validate_simulation_config(invalid)


def test_best_of_length_changes_series_probability() -> None:
    bo3 = adjusted_series_probability(0.75, best_of=3, reference_best_of=3)
    bo5 = adjusted_series_probability(0.75, best_of=5, reference_best_of=3)
    bo7 = adjusted_series_probability(0.75, best_of=7, reference_best_of=3)

    assert bo3 == pytest.approx(0.75)
    assert 0.75 < bo5 < bo7 < 1.0
    assert adjusted_series_probability(0.5, 7, 3) == pytest.approx(0.5)


def test_six_team_bracket_routes_winners_and_losers(season18_format: dict) -> None:
    seeds = np.arange(6)
    probabilities = np.full((6, 6), 0.5)
    for team_a in range(6):
        for team_b in range(6):
            if team_a != team_b:
                probabilities[team_a, team_b] = float(team_a < team_b)

    champion, finalists, trace = simulate_playoff_bracket(
        seeds,
        probabilities,
        np.random.default_rng(18),
        season18_format["playoffs"],
        include_trace=True,
    )
    matches = {match["match_id"]: match for match in trace}

    assert champion == 0
    assert finalists == (0, 1)
    assert matches["play_in_1"]["team_a"] == 2
    assert matches["play_in_1"]["team_b"] == 5
    assert matches["play_in_2"]["team_a"] == 3
    assert matches["play_in_2"]["team_b"] == 4
    assert matches["upper_semifinal_1"]["team_b"] == 2
    assert matches["upper_semifinal_2"]["team_b"] == 3
    assert matches["lower_semifinal"]["team_a"] == 2
    assert matches["lower_semifinal"]["team_b"] == 3
    assert matches["lower_final"]["team_a"] == 1
    assert matches["lower_final"]["team_b"] == 2
    assert {5, 4}.isdisjoint(
        {
            match[side]
            for match in trace[2:]
            for side in ("team_a", "team_b")
        }
    )


def test_future_season_snapshot_ids_are_not_hard_coded() -> None:
    schedule = pd.read_csv(PROJECT_ROOT / "data" / "season18" / "schedule_results.csv")
    schedule["season"] = 19
    schedule["scheduled_at"] = pd.to_datetime(schedule["scheduled_at"], utc=True).dt.tz_convert(
        "Asia/Jakarta"
    )

    windows = build_live_prediction_windows(schedule)

    assert windows["snapshot_id"].tolist() == ["S19_PRE", "S19_W01", "S19_W02", "S19_W03"]
