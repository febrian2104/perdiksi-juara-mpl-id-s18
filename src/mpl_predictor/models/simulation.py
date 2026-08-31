import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json
from mpl_predictor.analysis.prediction_policy import _match_completion_dates
from mpl_predictor.features.elo import EloTracker
from mpl_predictor.features.matches import _difference, _team_state
from mpl_predictor.features.snapshots import _append_match_history
from mpl_predictor.models.final import predict_match_probability

STATE_DIFFERENCE_FEATURES = (
    "current_matches",
    "current_match_win_rate",
    "current_game_win_rate",
    "current_game_diff_per_match",
    "current_form_3",
    "current_form_5",
    "current_sos_elo_avg",
    "prior_regular_match_win_rate",
    "prior_regular_game_win_rate",
    "prior_regular_game_diff_per_match",
    "prior_3_season_match_win_rate",
    "prior_3_season_game_win_rate",
    "prior_3_season_game_diff_per_match",
)


def load_simulation_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "simulation_version",
        "season",
        "iterations",
        "random_seed",
        "regular_season",
        "playoffs",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Simulation config is missing: {', '.join(sorted(missing))}")
    return config


def _naive_date(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Jakarta").tz_localize(None)
    return timestamp.normalize()


def _replay_historical_state(
    tables: dict[str, pd.DataFrame], feature_config: dict[str, Any]
) -> tuple[EloTracker, dict[str, list[dict[str, Any]]]]:
    elo = feature_config["elo"]
    tracker = EloTracker(
        initial_rating=float(elo["initial_rating"]),
        k_factor=float(elo["k_factor"]),
        scale=float(elo["scale"]),
        season_carryover=float(elo["season_carryover"]),
    )
    history: dict[str, list[dict[str, Any]]] = {}
    matches = _match_completion_dates(tables["matches"], tables["games"])
    matches = matches.loc[matches["season"].ge(4)]
    teams = tables["teams"].loc[tables["teams"]["season"].ge(4)]
    for season in sorted(int(value) for value in matches["season"].unique()):
        active_slots = (
            teams.loc[teams["season"].eq(season), "franchise_slot_id"].astype(str).tolist()
        )
        tracker.regress_for_new_season(active_slots)
        season_matches = matches.loc[matches["season"].eq(season)].sort_values(
            ["completion_date", "match_id"]
        )
        for match in season_matches.itertuples(index=False):
            _append_match_history(match, tracker, history)
    return tracker, history


def _match_feature_record(
    slot_a: str,
    slot_b: str,
    tracker: EloTracker,
    history: dict[str, list[dict[str, Any]]],
    scheduled_at: Any | None = None,
) -> dict[str, float | None]:
    state_a = _team_state(slot_a, 18, history)
    state_b = _team_state(slot_b, 18, history)
    rating_a = tracker.rating(slot_a)
    rating_b = tracker.rating(slot_b)
    record: dict[str, float | None] = {"elo_rating_diff": rating_a - rating_b}
    for key in STATE_DIFFERENCE_FEATURES:
        record[f"{key}_diff"] = _difference(state_a[key], state_b[key])
    if scheduled_at is None:
        record["rest_days_diff"] = None
    else:
        scheduled_date = _naive_date(scheduled_at)
        rest_a = (
            (scheduled_date - _naive_date(state_a["latest_match_completion_date"])).days
            if pd.notna(state_a["latest_match_completion_date"])
            else None
        )
        rest_b = (
            (scheduled_date - _naive_date(state_b["latest_match_completion_date"])).days
            if pd.notna(state_b["latest_match_completion_date"])
            else None
        )
        record["rest_days_diff"] = _difference(rest_a, rest_b)
    return record


def _live_match(row: Any) -> SimpleNamespace:
    return SimpleNamespace(
        season=18,
        stage="regular_season",
        match_id=str(row.match_id),
        completion_date=_naive_date(row.scheduled_at),
        team_a_franchise_slot_id=str(row.team_a_franchise_slot_id),
        team_b_franchise_slot_id=str(row.team_b_franchise_slot_id),
        winner_side=str(row.winner_side),
        team_a_score=int(row.team_a_score),
        team_b_score=int(row.team_b_score),
    )


def build_season18_match_probabilities(
    tables: dict[str, pd.DataFrame],
    schedule: pd.DataFrame,
    teams: pd.DataFrame,
    feature_config: dict[str, Any],
    artifact: dict[str, Any],
) -> tuple[pd.DataFrame, EloTracker, dict[str, list[dict[str, Any]]]]:
    """Replay S4-S17, incorporate published S18 results, and price remaining matches."""
    tracker, history = _replay_historical_state(tables, feature_config)
    active_slots = teams["franchise_slot_id"].astype(str).tolist()
    tracker.regress_for_new_season(active_slots)
    records = []
    completed = schedule.loc[schedule["status"].eq("completed")].sort_values(
        ["scheduled_at", "official_match_id"]
    )
    for row in completed.itertuples(index=False):
        features = _match_feature_record(
            str(row.team_a_franchise_slot_id),
            str(row.team_b_franchise_slot_id),
            tracker,
            history,
            row.scheduled_at,
        )
        probability = predict_match_probability(artifact, features)
        records.append(
            {
                **row._asdict(),
                "team_a_win_probability": probability,
                "probability_basis": "historical_pre_match_state",
            }
        )
        _append_match_history(_live_match(row), tracker, history)

    remaining = schedule.loc[schedule["status"].eq("scheduled")].sort_values(
        ["scheduled_at", "official_match_id"]
    )
    for row in remaining.itertuples(index=False):
        features = _match_feature_record(
            str(row.team_a_franchise_slot_id),
            str(row.team_b_franchise_slot_id),
            tracker,
            history,
            row.scheduled_at,
        )
        records.append(
            {
                **row._asdict(),
                "team_a_win_probability": predict_match_probability(artifact, features),
                "probability_basis": "current_as_of_state",
            }
        )
    probabilities = pd.DataFrame.from_records(records).sort_values(
        ["scheduled_at", "official_match_id"]
    )
    probabilities["predicted_winner_team_id"] = np.where(
        probabilities["team_a_win_probability"].ge(0.5),
        probabilities["team_a_id"],
        probabilities["team_b_id"],
    )
    return probabilities.reset_index(drop=True), tracker, history


def _regular_sweep_probability(historical_matches: pd.DataFrame) -> float:
    regular = historical_matches.loc[
        historical_matches["season"].ge(4)
        & historical_matches["stage"].eq("regular_season")
        & historical_matches["best_of"].eq(3)
    ]
    losing_scores = regular[["team_a_score", "team_b_score"]].min(axis=1)
    return float(losing_scores.eq(0).mean())


def _base_standings(schedule: pd.DataFrame, team_ids: list[str]) -> dict[str, np.ndarray]:
    index = {team_id: position for position, team_id in enumerate(team_ids)}
    stats = {
        "match_wins": np.zeros(len(team_ids), dtype=np.int16),
        "match_losses": np.zeros(len(team_ids), dtype=np.int16),
        "game_wins": np.zeros(len(team_ids), dtype=np.int16),
        "game_losses": np.zeros(len(team_ids), dtype=np.int16),
    }
    for row in schedule.loc[schedule["status"].eq("completed")].itertuples(index=False):
        a = index[str(row.team_a_id)]
        b = index[str(row.team_b_id)]
        score_a = int(row.team_a_score)
        score_b = int(row.team_b_score)
        stats["game_wins"][a] += score_a
        stats["game_losses"][a] += score_b
        stats["game_wins"][b] += score_b
        stats["game_losses"][b] += score_a
        winner, loser = (a, b) if score_a > score_b else (b, a)
        stats["match_wins"][winner] += 1
        stats["match_losses"][loser] += 1
    return stats


def _current_standings(schedule: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    team_ids = teams["team_id"].astype(str).tolist()
    names = teams.set_index("team_id")["team_name"].to_dict()
    stats = _base_standings(schedule, team_ids)
    game_diff = stats["game_wins"] - stats["game_losses"]
    order = np.lexsort((-stats["game_wins"], -game_diff, -stats["match_wins"]))
    return pd.DataFrame(
        {
            "current_rank": np.arange(1, len(team_ids) + 1),
            "team_id": [team_ids[index] for index in order],
            "team_name": [names[team_ids[index]] for index in order],
            "match_wins": stats["match_wins"][order],
            "match_losses": stats["match_losses"][order],
            "game_wins": stats["game_wins"][order],
            "game_losses": stats["game_losses"][order],
            "game_differential": game_diff[order],
        }
    )


def _pair_probability_matrix(
    team_ids: list[str],
    slot_lookup: dict[str, str],
    tracker: EloTracker,
    history: dict[str, list[dict[str, Any]]],
    artifact: dict[str, Any],
) -> np.ndarray:
    matrix = np.full((len(team_ids), len(team_ids)), 0.5, dtype=float)
    for a in range(len(team_ids)):
        for b in range(a + 1, len(team_ids)):
            features = _match_feature_record(
                slot_lookup[team_ids[a]], slot_lookup[team_ids[b]], tracker, history
            )
            probability = predict_match_probability(artifact, features)
            matrix[a, b] = probability
            matrix[b, a] = 1.0 - probability
    return matrix


def _sample_series(
    a: int, b: int, probabilities: np.ndarray, rng: np.random.Generator
) -> tuple[int, int]:
    return (a, b) if rng.random() < probabilities[a, b] else (b, a)


def _simulate_playoffs(
    seeds: np.ndarray, probabilities: np.ndarray, rng: np.random.Generator
) -> tuple[int, tuple[int, int]]:
    winner_36, _ = _sample_series(int(seeds[2]), int(seeds[5]), probabilities, rng)
    winner_45, _ = _sample_series(int(seeds[3]), int(seeds[4]), probabilities, rng)
    winner_upper_1, loser_upper_1 = _sample_series(int(seeds[0]), winner_36, probabilities, rng)
    winner_upper_2, loser_upper_2 = _sample_series(int(seeds[1]), winner_45, probabilities, rng)
    winner_lower_semifinal, _ = _sample_series(loser_upper_1, loser_upper_2, probabilities, rng)
    winner_upper_final, loser_upper_final = _sample_series(
        winner_upper_1, winner_upper_2, probabilities, rng
    )
    winner_lower_final, _ = _sample_series(
        loser_upper_final, winner_lower_semifinal, probabilities, rng
    )
    champion, _ = _sample_series(winner_upper_final, winner_lower_final, probabilities, rng)
    return champion, (winner_upper_final, winner_lower_final)


def simulate_season18(
    tables: dict[str, pd.DataFrame],
    schedule: pd.DataFrame,
    teams: pd.DataFrame,
    match_probabilities: pd.DataFrame,
    tracker: EloTracker,
    history: dict[str, list[dict[str, Any]]],
    artifact: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate remaining regular season and the recent six-team playoff bracket."""
    iterations = int(config["iterations"])
    rng = np.random.default_rng(int(config["random_seed"]))
    team_ids = teams["team_id"].astype(str).tolist()
    team_names = teams.set_index("team_id")["team_name"].to_dict()
    team_index = {team_id: index for index, team_id in enumerate(team_ids)}
    slot_lookup = teams.set_index("team_id")["franchise_slot_id"].astype(str).to_dict()
    base = _base_standings(schedule, team_ids)
    remaining = match_probabilities.loc[match_probabilities["status"].eq("scheduled")]
    remaining_rows = [
        (
            team_index[str(row.team_a_id)],
            team_index[str(row.team_b_id)],
            float(row.team_a_win_probability),
        )
        for row in remaining.itertuples(index=False)
    ]
    pair_probabilities = _pair_probability_matrix(team_ids, slot_lookup, tracker, history, artifact)
    sweep_probability = _regular_sweep_probability(tables["matches"])
    team_count = len(team_ids)
    rank_sum = np.zeros(team_count, dtype=np.int64)
    playoff_count = np.zeros(team_count, dtype=np.int64)
    champion_count = np.zeros(team_count, dtype=np.int64)
    grand_final_count = np.zeros(team_count, dtype=np.int64)
    seed_counts = np.zeros((team_count, 6), dtype=np.int64)

    for _ in range(iterations):
        match_wins = base["match_wins"].copy()
        match_losses = base["match_losses"].copy()
        game_wins = base["game_wins"].copy()
        game_losses = base["game_losses"].copy()
        for a, b, probability in remaining_rows:
            a_wins = rng.random() < probability
            winner, loser = (a, b) if a_wins else (b, a)
            loser_score = 0 if rng.random() < sweep_probability else 1
            match_wins[winner] += 1
            match_losses[loser] += 1
            game_wins[winner] += 2
            game_losses[winner] += loser_score
            game_wins[loser] += loser_score
            game_losses[loser] += 2
        game_diff = game_wins - game_losses
        random_tiebreak = rng.random(team_count)
        seeds = np.lexsort((random_tiebreak, -game_wins, -game_diff, -match_wins))
        ranks = np.empty(team_count, dtype=np.int16)
        ranks[seeds] = np.arange(1, team_count + 1)
        rank_sum += ranks
        top_six = seeds[:6]
        playoff_count[top_six] += 1
        for seed_index, team in enumerate(top_six):
            seed_counts[team, seed_index] += 1
        champion, finalists = _simulate_playoffs(top_six, pair_probabilities, rng)
        champion_count[champion] += 1
        grand_final_count[list(finalists)] += 1

    records = []
    current = _current_standings(schedule, teams).set_index("team_id")
    for index, team_id in enumerate(team_ids):
        record = {
            "season": 18,
            "as_of": str(schedule["observed_at"].iloc[0]),
            "team_id": team_id,
            "team_name": team_names[team_id],
            "current_rank": int(current.loc[team_id, "current_rank"]),
            "current_match_wins": int(current.loc[team_id, "match_wins"]),
            "current_match_losses": int(current.loc[team_id, "match_losses"]),
            "expected_regular_rank": rank_sum[index] / iterations,
            "regular_first_probability": seed_counts[index, 0] / iterations,
            "playoff_probability": playoff_count[index] / iterations,
            "grand_final_probability": grand_final_count[index] / iterations,
            "champion_probability": champion_count[index] / iterations,
        }
        for seed_index in range(6):
            record[f"seed_{seed_index + 1}_probability"] = (
                seed_counts[index, seed_index] / iterations
            )
        records.append(record)
    result = pd.DataFrame.from_records(records).sort_values(
        ["champion_probability", "playoff_probability", "team_id"],
        ascending=[False, False, True],
    )
    result["champion_rank"] = np.arange(1, len(result) + 1)

    champion_sum = float(result["champion_probability"].sum())
    playoff_sum = float(result["playoff_probability"].sum())
    seed_sums = [float(result[f"seed_{index}_probability"].sum()) for index in range(1, 7)]
    tolerance = 1e-9
    checks = [
        {
            "check_id": "champion_probabilities_sum_to_one",
            "status": "pass" if abs(champion_sum - 1.0) < tolerance else "fail",
            "value": champion_sum,
        },
        {
            "check_id": "playoff_probabilities_sum_to_six",
            "status": "pass" if abs(playoff_sum - 6.0) < tolerance else "fail",
            "value": playoff_sum,
        },
        {
            "check_id": "each_seed_probabilities_sum_to_one",
            "status": "pass"
            if all(abs(value - 1.0) < tolerance for value in seed_sums)
            else "fail",
            "value": seed_sums,
        },
    ]
    report = {
        "report_version": "1.0",
        "simulation_version": config["simulation_version"],
        "season": 18,
        "as_of": str(schedule["observed_at"].iloc[0]),
        "iterations": iterations,
        "random_seed": int(config["random_seed"]),
        "input_scope": {
            "team_count": team_count,
            "regular_match_count": len(schedule),
            "completed_match_count": int(schedule["status"].eq("completed").sum()),
            "simulated_regular_match_count": len(remaining),
        },
        "format": {
            "regular_season": config["regular_season"],
            "playoffs": config["playoffs"],
            "playoff_bracket_basis": (
                "Dikonfigurasi mengikuti struktur delapan seri yang digunakan pada S15-S17."
            ),
        },
        "model": {
            "match_model": artifact["model_name"],
            "training_seasons": [
                artifact["training_season_min"],
                artifact["training_season_max"],
            ],
            "regular_series_score_sweep_probability": sweep_probability,
            "remaining_match_probabilities": (
                "Frozen pada state as-of; tidak dilatih ulang per iterasi."
            ),
            "roster_usage": (
                "Roster bertanggal terintegrasi, tetapi bukan kolom fitur model match final v1."
            ),
        },
        "validation": {
            "blocking_issue_count": sum(item["status"] == "fail" for item in checks),
            "checks": checks,
        },
        "current_standings": dataframe_records(_current_standings(schedule, teams)),
        "predictions": dataframe_records(result),
    }
    return result.reset_index(drop=True), report


def write_simulation_outputs(
    simulation: pd.DataFrame,
    match_probabilities: pd.DataFrame,
    report: dict[str, Any],
    simulation_path: Path,
    match_probability_path: Path,
    report_path: Path,
) -> None:
    simulation_path.parent.mkdir(parents=True, exist_ok=True)
    simulation.to_parquet(simulation_path, index=False)
    match_probabilities.to_parquet(match_probability_path, index=False)
    write_json(report, report_path)
