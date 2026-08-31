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
from mpl_predictor.models.online_learning import (
    OnlineTemperatureLearner,
    adapt_probability,
)
from mpl_predictor.models.tournament import (
    simulate_playoff_bracket,
    validate_simulation_config,
)

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
    validate_simulation_config(config)
    return config


def _target_season(
    schedule: pd.DataFrame,
    teams: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> int:
    season_values = pd.to_numeric(schedule["season"], errors="raise").dropna().unique()
    if len(season_values) != 1:
        raise ValueError("Live schedule must contain exactly one season.")
    target_season = int(season_values[0])
    if teams is not None:
        team_seasons = pd.to_numeric(teams["season"], errors="raise").dropna().unique()
        if len(team_seasons) != 1 or int(team_seasons[0]) != target_season:
            raise ValueError("Live teams and schedule must refer to the same single season.")
    if config is not None and int(config["season"]) != target_season:
        raise ValueError(
            "Simulation config season does not match live data: "
            f"{config['season']} != {target_season}."
        )
    return target_season


def _naive_date(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Jakarta").tz_localize(None)
    return timestamp.normalize()


def _replay_historical_state(
    tables: dict[str, pd.DataFrame], feature_config: dict[str, Any], target_season: int
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
    matches = matches.loc[matches["season"].ge(4) & matches["season"].lt(target_season)]
    teams = tables["teams"].loc[
        tables["teams"]["season"].ge(4) & tables["teams"]["season"].lt(target_season)
    ]
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
    target_season: int,
    tracker: EloTracker,
    history: dict[str, list[dict[str, Any]]],
    scheduled_at: Any | None = None,
) -> dict[str, float | None]:
    state_a = _team_state(slot_a, target_season, history)
    state_b = _team_state(slot_b, target_season, history)
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
        season=int(row.season),
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
    """Replay prior seasons, incorporate published live results, and price remaining matches."""
    target_season = _target_season(schedule, teams)
    tracker, history = _replay_historical_state(tables, feature_config, target_season)
    active_slots = teams["franchise_slot_id"].astype(str).tolist()
    tracker.regress_for_new_season(active_slots)
    online_config = feature_config.get("online_learning", {})
    learner = OnlineTemperatureLearner.from_config(online_config)
    learning_method = (
        str(online_config.get("method", "regularized_logit_temperature"))
        if learner.enabled
        else "disabled"
    )
    records = []
    completed = schedule.loc[schedule["status"].eq("completed")].sort_values(
        ["scheduled_at", "official_match_id"]
    )
    for row in completed.itertuples(index=False):
        features = _match_feature_record(
            str(row.team_a_franchise_slot_id),
            str(row.team_b_franchise_slot_id),
            target_season,
            tracker,
            history,
            row.scheduled_at,
        )
        base_probability = predict_match_probability(artifact, features)
        probability = learner.predict(base_probability)
        observations_before = learner.observation_count
        scale_before = learner.scale
        actual_team_a_win = float(str(row.winner_team_id) == str(row.team_a_id))
        error_signal = actual_team_a_win - probability
        scale_after = learner.update(base_probability, actual_team_a_win)
        records.append(
            {
                **row._asdict(),
                **features,
                "base_team_a_win_probability": base_probability,
                "team_a_win_probability": probability,
                "probability_basis": "historical_pre_match_state",
                "online_learning_method": learning_method,
                "online_learning_observation_count": observations_before,
                "online_learning_scale": scale_before,
                "online_learning_error_signal": error_signal,
                "online_learning_scale_after_update": scale_after,
                "online_learning_update_applied": learner.enabled,
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
            target_season,
            tracker,
            history,
            row.scheduled_at,
        )
        base_probability = predict_match_probability(artifact, features)
        records.append(
            {
                **row._asdict(),
                **features,
                "base_team_a_win_probability": base_probability,
                "team_a_win_probability": learner.predict(base_probability),
                "probability_basis": "current_as_of_state",
                "online_learning_method": learning_method,
                "online_learning_observation_count": learner.observation_count,
                "online_learning_scale": learner.scale,
                "online_learning_error_signal": np.nan,
                "online_learning_scale_after_update": learner.scale,
                "online_learning_update_applied": False,
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
    probabilities["base_predicted_winner_team_id"] = np.where(
        probabilities["base_team_a_win_probability"].ge(0.5),
        probabilities["team_a_id"],
        probabilities["team_b_id"],
    )
    evaluated = probabilities["status"].eq("completed") & probabilities["winner_team_id"].notna()
    prediction_correct = pd.Series(pd.NA, index=probabilities.index, dtype="boolean")
    prediction_correct.loc[evaluated] = probabilities.loc[evaluated, "predicted_winner_team_id"].eq(
        probabilities.loc[evaluated, "winner_team_id"]
    )
    probabilities["prediction_correct"] = prediction_correct
    base_prediction_correct = pd.Series(pd.NA, index=probabilities.index, dtype="boolean")
    base_prediction_correct.loc[evaluated] = probabilities.loc[
        evaluated, "base_predicted_winner_team_id"
    ].eq(probabilities.loc[evaluated, "winner_team_id"])
    probabilities["base_prediction_correct"] = base_prediction_correct
    probabilities["accuracy_status"] = "pending_result"
    probabilities.loc[evaluated & prediction_correct.fillna(False), "accuracy_status"] = "correct"
    probabilities.loc[evaluated & ~prediction_correct.fillna(False), "accuracy_status"] = (
        "incorrect"
    )
    probabilities["result_update_status"] = np.where(
        evaluated,
        "incorporated_after_pre_match_prediction",
        "awaiting_result",
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


def _current_standings(
    schedule: pd.DataFrame,
    teams: pd.DataFrame,
    ranking_rules: list[str] | None = None,
) -> pd.DataFrame:
    team_ids = teams["team_id"].astype(str).tolist()
    names = teams.set_index("team_id")["team_name"].to_dict()
    stats = _base_standings(schedule, team_ids)
    game_diff = stats["game_wins"] - stats["game_losses"]
    ranking_rules = ranking_rules or [
        "match_wins_desc",
        "game_differential_desc",
        "game_wins_desc",
        "random_tiebreak",
    ]
    order = _ranking_order(
        ranking_rules,
        stats["match_wins"],
        stats["match_losses"],
        stats["game_wins"],
        stats["game_losses"],
        random_tiebreak=np.arange(len(team_ids), dtype=float),
    )
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
    target_season: int,
    tracker: EloTracker,
    history: dict[str, list[dict[str, Any]]],
    artifact: dict[str, Any],
    online_learning_scale: float = 1.0,
) -> np.ndarray:
    matrix = np.full((len(team_ids), len(team_ids)), 0.5, dtype=float)
    for a in range(len(team_ids)):
        for b in range(a + 1, len(team_ids)):
            features = _match_feature_record(
                slot_lookup[team_ids[a]],
                slot_lookup[team_ids[b]],
                target_season,
                tracker,
                history,
            )
            probability = adapt_probability(
                predict_match_probability(artifact, features), online_learning_scale
            )
            matrix[a, b] = probability
            matrix[b, a] = 1.0 - probability
    return matrix


def _ranking_order(
    ranking_rules: list[str],
    match_wins: np.ndarray,
    match_losses: np.ndarray,
    game_wins: np.ndarray,
    game_losses: np.ndarray,
    rng: np.random.Generator | None = None,
    random_tiebreak: np.ndarray | None = None,
) -> np.ndarray:
    game_differential = game_wins - game_losses
    values = {
        "match_wins_desc": -match_wins,
        "match_losses_asc": match_losses,
        "game_differential_desc": -game_differential,
        "game_wins_desc": -game_wins,
        "random_tiebreak": (
            random_tiebreak
            if random_tiebreak is not None
            else rng.random(len(match_wins))
            if rng is not None
            else np.arange(len(match_wins), dtype=float)
        ),
    }
    return np.lexsort(tuple(values[rule] for rule in reversed(ranking_rules)))


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
    """Simulate a configured live season and its declarative playoff bracket."""
    format_summary = validate_simulation_config(config)
    target_season = _target_season(schedule, teams, config)
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
    completed_probabilities = match_probabilities.loc[match_probabilities["status"].eq("completed")]
    if not remaining.empty and "online_learning_scale" in remaining:
        online_learning_scale = float(remaining["online_learning_scale"].iloc[0])
    elif (
        not completed_probabilities.empty
        and "online_learning_scale_after_update" in completed_probabilities
    ):
        online_learning_scale = float(
            completed_probabilities.sort_values(["scheduled_at", "official_match_id"])[
                "online_learning_scale_after_update"
            ].iloc[-1]
        )
    else:
        online_learning_scale = 1.0
    if not remaining.empty and "online_learning_method" in remaining:
        online_learning_method = str(remaining["online_learning_method"].iloc[0])
    elif not completed_probabilities.empty and "online_learning_method" in completed_probabilities:
        online_learning_method = str(completed_probabilities["online_learning_method"].iloc[-1])
    else:
        online_learning_method = "disabled"
    pair_probabilities = _pair_probability_matrix(
        team_ids,
        slot_lookup,
        target_season,
        tracker,
        history,
        artifact,
        online_learning_scale,
    )
    sweep_probability = _regular_sweep_probability(tables["matches"])
    team_count = len(team_ids)
    rank_sum = np.zeros(team_count, dtype=np.int64)
    playoff_count = np.zeros(team_count, dtype=np.int64)
    champion_count = np.zeros(team_count, dtype=np.int64)
    grand_final_count = np.zeros(team_count, dtype=np.int64)
    playoff_team_count = int(config["regular_season"]["playoff_team_count"])
    if playoff_team_count > team_count:
        raise ValueError(
            f"Configured playoff team count {playoff_team_count} exceeds {team_count} teams."
        )
    seed_counts = np.zeros((team_count, playoff_team_count), dtype=np.int64)
    ranking_rules = list(config["regular_season"]["ranking_order"])

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
        seeds = _ranking_order(ranking_rules, match_wins, match_losses, game_wins, game_losses, rng)
        ranks = np.empty(team_count, dtype=np.int16)
        ranks[seeds] = np.arange(1, team_count + 1)
        rank_sum += ranks
        playoff_seeds = seeds[:playoff_team_count]
        playoff_count[playoff_seeds] += 1
        for seed_index, team in enumerate(playoff_seeds):
            seed_counts[team, seed_index] += 1
        champion, finalists, _ = simulate_playoff_bracket(
            playoff_seeds, pair_probabilities, rng, config["playoffs"]
        )
        champion_count[champion] += 1
        grand_final_count[list(finalists)] += 1

    records = []
    current = _current_standings(schedule, teams, ranking_rules).set_index("team_id")
    for index, team_id in enumerate(team_ids):
        record = {
            "season": target_season,
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
        for seed_index in range(playoff_team_count):
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
    seed_sums = [
        float(result[f"seed_{index}_probability"].sum())
        for index in range(1, playoff_team_count + 1)
    ]
    tolerance = 1e-9
    checks = [
        {
            "check_id": "champion_probabilities_sum_to_one",
            "status": "pass" if abs(champion_sum - 1.0) < tolerance else "fail",
            "value": champion_sum,
        },
        {
            "check_id": "playoff_probabilities_sum_to_configured_team_count",
            "status": ("pass" if abs(playoff_sum - playoff_team_count) < tolerance else "fail"),
            "value": playoff_sum,
            "expected": playoff_team_count,
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
        "season": target_season,
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
            "format_confirmation": config["format_confirmation"],
            "validated_format_summary": format_summary,
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
            "playoff_series_probability": (
                "Probabilitas seri referensi dikonversi menjadi peluang per game, lalu "
                "dihitung ulang sesuai best-of setiap match bracket."
            ),
            "roster_usage": (
                "Roster bertanggal terintegrasi, tetapi bukan kolom fitur model match final v1."
            ),
            "online_learning": {
                "method": online_learning_method,
                "completed_result_count": int(schedule["status"].eq("completed").sum()),
                "final_confidence_scale": online_learning_scale,
                "timing": "Setiap outcome digunakan hanya setelah probabilitas pre-match.",
            },
        },
        "validation": {
            "blocking_issue_count": sum(item["status"] == "fail" for item in checks),
            "checks": checks,
        },
        "current_standings": dataframe_records(_current_standings(schedule, teams, ranking_rules)),
        "predictions": dataframe_records(result),
    }
    return result.reset_index(drop=True), report


# Generic aliases for future seasons. Existing S18 names remain API-compatible.
build_live_match_probabilities = build_season18_match_probabilities
simulate_live_season = simulate_season18


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
