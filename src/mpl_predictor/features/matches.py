from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json
from mpl_predictor.analysis.prediction_policy import _match_completion_dates
from mpl_predictor.features.elo import EloTracker
from mpl_predictor.features.snapshots import (
    _actual_score,
    _append_match_history,
    _form_score,
    _summarize,
)


def _team_state(
    slot: str,
    season: int,
    history: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    records = history.get(slot, [])
    current = [record for record in records if record["season"] == season]
    prior_regular = [
        record
        for record in records
        if record["season"] == season - 1 and record["stage"] == "regular_season"
    ]
    prior_rolling = [record for record in records if season - 3 <= record["season"] < season]
    current_summary = _summarize(current)
    prior_summary = _summarize(prior_regular)
    rolling_summary = _summarize(prior_rolling)
    latest_date = max(
        (record["completion_date"] for record in records),
        default=pd.NaT,
    )
    return {
        "rating_history_matches": len(records),
        "current_matches": current_summary["matches"],
        "current_match_win_rate": current_summary["match_win_rate"],
        "current_game_win_rate": current_summary["game_win_rate"],
        "current_game_diff_per_match": current_summary["game_diff_per_match"],
        "current_form_3": _form_score(current, 3),
        "current_form_5": _form_score(current, 5),
        "current_sos_elo_avg": (
            sum(float(record["opponent_elo_before"]) for record in current) / len(current)
            if current
            else None
        ),
        "prior_regular_match_win_rate": prior_summary["match_win_rate"],
        "prior_regular_game_win_rate": prior_summary["game_win_rate"],
        "prior_regular_game_diff_per_match": prior_summary["game_diff_per_match"],
        "prior_3_season_match_win_rate": rolling_summary["match_win_rate"],
        "prior_3_season_game_win_rate": rolling_summary["game_win_rate"],
        "prior_3_season_game_diff_per_match": rolling_summary["game_diff_per_match"],
        "latest_match_completion_date": latest_date,
    }


def _difference(value_a: Any, value_b: Any) -> float | None:
    if value_a is None or value_b is None or pd.isna(value_a) or pd.isna(value_b):
        return None
    return float(value_a) - float(value_b)


def build_match_features(
    tables: dict[str, pd.DataFrame],
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    """Create pre-match, side-symmetric features for every franchise-era match."""
    matches = _match_completion_dates(tables["matches"], tables["games"])
    matches = matches.loc[matches["season"].ge(4)].copy()
    teams = tables["teams"].loc[tables["teams"]["season"].ge(4)]
    elo_config = feature_config["elo"]
    tracker = EloTracker(
        initial_rating=float(elo_config["initial_rating"]),
        k_factor=float(elo_config["k_factor"]),
        scale=float(elo_config["scale"]),
        season_carryover=float(elo_config["season_carryover"]),
    )
    history: dict[str, list[dict[str, Any]]] = {}
    records = []

    for season in sorted(int(value) for value in matches["season"].unique()):
        active_slots = (
            teams.loc[teams["season"].eq(season), "franchise_slot_id"].astype(str).tolist()
        )
        tracker.regress_for_new_season(active_slots)
        season_matches = matches.loc[matches["season"].eq(season)].sort_values(
            ["completion_date", "match_id"]
        )
        for match in season_matches.itertuples(index=False):
            slot_a = str(match.team_a_franchise_slot_id)
            slot_b = str(match.team_b_franchise_slot_id)
            state_a = _team_state(slot_a, season, history)
            state_b = _team_state(slot_b, season, history)
            rating_a = tracker.rating(slot_a)
            rating_b = tracker.rating(slot_b)
            rest_a = (
                (match.completion_date - state_a["latest_match_completion_date"]).days
                if pd.notna(state_a["latest_match_completion_date"])
                else None
            )
            rest_b = (
                (match.completion_date - state_b["latest_match_completion_date"]).days
                if pd.notna(state_b["latest_match_completion_date"])
                else None
            )
            record = {
                "feature_version": feature_config["feature_version"],
                "season": season,
                "stage": str(match.stage),
                "week": match.week,
                "round": match.round,
                "match_id": str(match.match_id),
                "completion_date": match.completion_date,
                "team_a_id": str(match.team_a_id),
                "team_b_id": str(match.team_b_id),
                "team_a_franchise_slot_id": slot_a,
                "team_b_franchise_slot_id": slot_b,
                "team_a_score": int(match.team_a_score),
                "team_b_score": int(match.team_b_score),
                "team_a_win": int(_actual_score(str(match.winner_side)) == 1.0),
                "elo_rating_a": rating_a,
                "elo_rating_b": rating_b,
                "elo_rating_diff": rating_a - rating_b,
                "elo_expected_team_a": tracker.expected_score(rating_a, rating_b),
                "latest_history_date_a": state_a["latest_match_completion_date"],
                "latest_history_date_b": state_b["latest_match_completion_date"],
                "rest_days_diff": _difference(rest_a, rest_b),
            }
            for key in (
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
            ):
                record[f"{key}_diff"] = _difference(state_a[key], state_b[key])
            records.append(record)
            _append_match_history(match, tracker, history)

    result = pd.DataFrame.from_records(records)
    result["season"] = result["season"].astype("Int64")
    result["week"] = result["week"].astype("Int64")
    result["team_a_win"] = result["team_a_win"].astype("Int64")
    return result.sort_values(["season", "completion_date", "match_id"]).reset_index(drop=True)


def build_match_feature_report(features: pd.DataFrame) -> dict[str, Any]:
    duplicate_matches = int(features["match_id"].duplicated().sum())
    missing_targets = int(features["team_a_win"].isna().sum())
    date_leaks = int(
        (
            features["latest_history_date_a"].gt(features["completion_date"])
            | features["latest_history_date_b"].gt(features["completion_date"])
        ).sum()
    )
    checks = [
        {
            "check_id": "unique_match_rows",
            "status": "pass" if duplicate_matches == 0 else "fail",
            "count": duplicate_matches,
        },
        {
            "check_id": "complete_binary_target",
            "status": "pass" if missing_targets == 0 else "fail",
            "count": missing_targets,
        },
        {
            "check_id": "history_not_after_match",
            "status": "pass" if date_leaks == 0 else "fail",
            "count": date_leaks,
        },
    ]
    feature_columns = [column for column in features if column.endswith("_diff")]
    missingness = pd.DataFrame(
        [
            {
                "feature": column,
                "missing_count": int(features[column].isna().sum()),
                "missing_pct": round(float(features[column].isna().mean() * 100), 2),
            }
            for column in feature_columns
        ]
    ).sort_values(["missing_pct", "feature"], ascending=[False, True])
    return {
        "report_version": "1.0",
        "match_row_count": len(features),
        "season_min": int(features["season"].min()),
        "season_max": int(features["season"].max()),
        "team_a_win_rate": round(float(features["team_a_win"].mean()), 6),
        "feature_columns": feature_columns,
        "blocking_issue_count": sum(check["status"] == "fail" for check in checks),
        "checks": checks,
        "feature_missingness": dataframe_records(missingness),
    }


def write_match_feature_outputs(
    features: pd.DataFrame,
    report: dict[str, Any],
    feature_path: Path,
    report_path: Path,
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    write_json(report, report_path)
