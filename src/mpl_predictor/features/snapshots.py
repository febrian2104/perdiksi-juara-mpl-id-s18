import json
from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json
from mpl_predictor.analysis.prediction_policy import _match_completion_dates
from mpl_predictor.features.elo import EloTracker
from mpl_predictor.features.roster import add_roster_features

IDENTIFIER_COLUMNS = [
    "feature_version",
    "snapshot_id",
    "season",
    "prediction_type",
    "completed_week",
    "feature_cutoff_date",
    "team_id",
    "team_name",
    "organization_id",
    "franchise_slot_id",
]
TARGET_COLUMNS = ["target_available", "champion"]

FEATURE_GROUPS = {
    "elo": [
        "elo_rating",
        "elo_rank",
        "elo_change_since_preseason",
        "elo_expected_vs_league_average",
    ],
    "prior_season_performance": [
        "prior_season_regular_matches",
        "prior_season_regular_match_win_rate",
        "prior_season_regular_game_win_rate",
        "prior_season_regular_game_diff_per_match",
        "prior_season_playoff_matches",
        "prior_season_playoff_match_win_rate",
        "prior_season_sos_opponent_match_win_rate",
        "prior_season_sos_adjusted_match_win_rate",
    ],
    "rolling_history": [
        "prior_3_season_matches",
        "prior_3_season_match_win_rate",
        "prior_3_season_game_win_rate",
        "prior_3_season_game_diff_per_match",
        "prior_3_season_sos_elo_avg",
    ],
    "current_performance": [
        "current_regular_matches",
        "current_regular_wins",
        "current_regular_losses",
        "current_regular_draws",
        "current_regular_match_win_rate",
        "current_regular_game_win_rate",
        "current_regular_game_differential",
        "current_regular_game_diff_per_match",
        "current_last_3_match_score",
        "current_last_5_match_score",
    ],
    "strength_of_schedule": [
        "current_sos_opponent_elo_avg",
        "current_sos_opponent_match_win_rate",
        "current_sos_adjusted_match_win_rate",
    ],
    "lagged_roster": [
        "lagged_roster_available",
        "lagged_roster_size",
        "lagged_roster_role_coverage",
        "lagged_roster_returning_count",
        "lagged_roster_continuity",
        "lagged_roster_avg_experience_seasons",
    ],
    "current_roster_temporal": [
        "current_roster_temporal_available",
        "current_roster_size_asof",
        "current_roster_role_coverage_asof",
        "current_roster_retained_count_asof",
        "current_roster_retained_share_asof",
        "current_roster_avg_experience_seasons_asof",
    ],
}


def load_feature_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"feature_version", "elo", "team_performance", "roster", "baseline"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Feature config is missing sections: {', '.join(missing)}")
    return config


def model_feature_columns() -> list[str]:
    return [column for columns in FEATURE_GROUPS.values() for column in columns]


def _actual_score(winner_side: str) -> float:
    if winner_side == "team_a":
        return 1.0
    if winner_side == "team_b":
        return 0.0
    if winner_side == "draw":
        return 0.5
    raise ValueError(f"Cannot update Elo from winner_side={winner_side!r}.")


def _summarize(records: list[dict[str, Any]]) -> dict[str, int | float | None]:
    matches = len(records)
    wins = sum(record["result"] == 1.0 for record in records)
    losses = sum(record["result"] == 0.0 for record in records)
    draws = sum(record["result"] == 0.5 for record in records)
    games_for = sum(int(record["games_for"]) for record in records)
    games_against = sum(int(record["games_against"]) for record in records)
    decided = wins + losses
    total_games = games_for + games_against
    return {
        "matches": matches,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "match_win_rate": wins / decided if decided else None,
        "game_win_rate": games_for / total_games if total_games else None,
        "game_differential": games_for - games_against,
        "game_diff_per_match": (games_for - games_against) / matches if matches else None,
    }


def _form_score(records: list[dict[str, Any]], window: int) -> float | None:
    recent = records[-window:]
    return sum(float(record["result"]) for record in recent) / len(recent) if recent else None


def _opponent_win_rate(
    records: list[dict[str, Any]],
    history: dict[str, list[dict[str, Any]]],
    season: int,
) -> float | None:
    opponent_rates = []
    for record in records:
        opponent_records = [
            item
            for item in history.get(record["opponent_id"], [])
            if item["season"] == season and item["stage"] == "regular_season"
        ]
        rate = _summarize(opponent_records)["match_win_rate"]
        if rate is not None:
            opponent_rates.append(float(rate))
    return sum(opponent_rates) / len(opponent_rates) if opponent_rates else None


def _team_snapshot_record(
    *,
    feature_version: str,
    window: pd.Series,
    team: pd.Series,
    history: dict[str, list[dict[str, Any]]],
    tracker: EloTracker,
    season_start_ratings: dict[str, float],
    elo_ranks: dict[str, float],
    league_average_rating: float,
    prior_seasons_window: int,
    target_lookup: dict[tuple[int, str], bool],
    target_seasons: set[int],
) -> dict[str, Any]:
    season = int(window["season"])
    slot = str(team["franchise_slot_id"])
    all_records = history.get(slot, [])
    current = [
        record
        for record in all_records
        if record["season"] == season and record["stage"] == "regular_season"
    ]
    prior_regular = [
        record
        for record in all_records
        if record["season"] == season - 1 and record["stage"] == "regular_season"
    ]
    prior_playoff = [
        record
        for record in all_records
        if record["season"] == season - 1 and record["stage"] == "playoffs"
    ]
    prior_rolling = [
        record
        for record in all_records
        if season - prior_seasons_window <= record["season"] < season
    ]
    current_summary = _summarize(current)
    prior_regular_summary = _summarize(prior_regular)
    prior_playoff_summary = _summarize(prior_playoff)
    prior_rolling_summary = _summarize(prior_rolling)
    current_opponent_win_rate = _opponent_win_rate(current, history, season)
    prior_opponent_win_rate = _opponent_win_rate(prior_regular, history, season - 1)
    rating = tracker.rating(slot)
    current_win_rate = current_summary["match_win_rate"]
    prior_win_rate = prior_regular_summary["match_win_rate"]
    target_available = season in target_seasons
    latest_match = max(
        (record["completion_date"] for record in all_records),
        default=pd.NaT,
    )

    return {
        "feature_version": feature_version,
        "snapshot_id": window["snapshot_id"],
        "season": season,
        "prediction_type": window["prediction_type"],
        "completed_week": window["completed_week"],
        "feature_cutoff_date": window["feature_cutoff_date"],
        "team_id": team["team_id"],
        "team_name": team["team_name"],
        "organization_id": team["organization_id"],
        "franchise_slot_id": slot,
        "target_available": target_available,
        "champion": target_lookup.get((season, str(team["team_id"]))) if target_available else None,
        "latest_match_completion_date_used": latest_match,
        "elo_rating": round(rating, 6),
        "elo_rank": elo_ranks[slot],
        "elo_change_since_preseason": round(rating - season_start_ratings[slot], 6),
        "elo_expected_vs_league_average": round(
            tracker.expected_score(rating, league_average_rating), 6
        ),
        "prior_season_regular_matches": prior_regular_summary["matches"],
        "prior_season_regular_match_win_rate": prior_regular_summary["match_win_rate"],
        "prior_season_regular_game_win_rate": prior_regular_summary["game_win_rate"],
        "prior_season_regular_game_diff_per_match": prior_regular_summary["game_diff_per_match"],
        "prior_season_playoff_matches": prior_playoff_summary["matches"],
        "prior_season_playoff_match_win_rate": prior_playoff_summary["match_win_rate"],
        "prior_season_sos_opponent_match_win_rate": prior_opponent_win_rate,
        "prior_season_sos_adjusted_match_win_rate": (
            float(prior_win_rate) - prior_opponent_win_rate
            if prior_win_rate is not None and prior_opponent_win_rate is not None
            else None
        ),
        "prior_3_season_matches": prior_rolling_summary["matches"],
        "prior_3_season_match_win_rate": prior_rolling_summary["match_win_rate"],
        "prior_3_season_game_win_rate": prior_rolling_summary["game_win_rate"],
        "prior_3_season_game_diff_per_match": prior_rolling_summary["game_diff_per_match"],
        "prior_3_season_sos_elo_avg": (
            sum(float(record["opponent_elo_before"]) for record in prior_rolling)
            / len(prior_rolling)
            if prior_rolling
            else None
        ),
        "current_regular_matches": current_summary["matches"],
        "current_regular_wins": current_summary["wins"],
        "current_regular_losses": current_summary["losses"],
        "current_regular_draws": current_summary["draws"],
        "current_regular_match_win_rate": current_summary["match_win_rate"],
        "current_regular_game_win_rate": current_summary["game_win_rate"],
        "current_regular_game_differential": current_summary["game_differential"],
        "current_regular_game_diff_per_match": current_summary["game_diff_per_match"],
        "current_last_3_match_score": _form_score(current, 3),
        "current_last_5_match_score": _form_score(current, 5),
        "current_sos_opponent_elo_avg": (
            sum(float(record["opponent_elo_before"]) for record in current) / len(current)
            if current
            else None
        ),
        "current_sos_opponent_match_win_rate": current_opponent_win_rate,
        "current_sos_adjusted_match_win_rate": (
            float(current_win_rate) - current_opponent_win_rate
            if current_win_rate is not None and current_opponent_win_rate is not None
            else None
        ),
    }


def _append_match_history(
    match: Any,
    tracker: EloTracker,
    history: dict[str, list[dict[str, Any]]],
) -> None:
    slot_a = str(match.team_a_franchise_slot_id)
    slot_b = str(match.team_b_franchise_slot_id)
    actual_a = _actual_score(str(match.winner_side))
    rating_a, rating_b = tracker.update(slot_a, slot_b, actual_a)
    base = {
        "season": int(match.season),
        "stage": str(match.stage),
        "match_id": str(match.match_id),
        "completion_date": match.completion_date,
    }
    history.setdefault(slot_a, []).append(
        {
            **base,
            "result": actual_a,
            "games_for": int(match.team_a_score),
            "games_against": int(match.team_b_score),
            "opponent_id": slot_b,
            "opponent_elo_before": rating_b,
        }
    )
    history.setdefault(slot_b, []).append(
        {
            **base,
            "result": 1.0 - actual_a,
            "games_for": int(match.team_b_score),
            "games_against": int(match.team_a_score),
            "opponent_id": slot_a,
            "opponent_elo_before": rating_a,
        }
    )


def _append_snapshot(
    records: list[dict[str, Any]],
    *,
    window: pd.Series,
    season_teams: pd.DataFrame,
    history: dict[str, list[dict[str, Any]]],
    tracker: EloTracker,
    season_start_ratings: dict[str, float],
    config: dict[str, Any],
    target_lookup: dict[tuple[int, str], bool],
    target_seasons: set[int],
) -> None:
    slots = season_teams["franchise_slot_id"].astype(str).tolist()
    ratings = pd.Series({slot: tracker.rating(slot) for slot in slots})
    elo_ranks = ratings.rank(method="average", ascending=False).to_dict()
    league_average = float(ratings.mean())
    for _, team in season_teams.sort_values("team_id").iterrows():
        records.append(
            _team_snapshot_record(
                feature_version=config["feature_version"],
                window=window,
                team=team,
                history=history,
                tracker=tracker,
                season_start_ratings=season_start_ratings,
                elo_ranks=elo_ranks,
                league_average_rating=league_average,
                prior_seasons_window=int(config["team_performance"]["prior_seasons_window"]),
                target_lookup=target_lookup,
                target_seasons=target_seasons,
            )
        )


def build_snapshot_features(
    tables: dict[str, pd.DataFrame],
    windows: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay historical matches and create one as-of feature row per team and snapshot."""
    matches = _match_completion_dates(tables["matches"], tables["games"])
    matches = matches.loc[matches["season"].ge(int(windows["season"].min()))].copy()
    teams = tables["teams"].loc[tables["teams"]["season"].isin(windows["season"])].copy()
    targets = tables["championships"]
    target_lookup = {
        (int(row.season), str(row.team_id)): bool(row.champion) for row in targets.itertuples()
    }
    target_seasons = set(int(value) for value in targets["season"].unique())
    elo_config = config["elo"]
    tracker = EloTracker(
        initial_rating=float(elo_config["initial_rating"]),
        k_factor=float(elo_config["k_factor"]),
        scale=float(elo_config["scale"]),
        season_carryover=float(elo_config["season_carryover"]),
    )
    history: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []

    for season in sorted(int(value) for value in windows["season"].unique()):
        season_teams = teams.loc[teams["season"].eq(season)]
        active_slots = season_teams["franchise_slot_id"].astype(str).tolist()
        tracker.regress_for_new_season(active_slots)
        season_start_ratings = {slot: tracker.rating(slot) for slot in active_slots}
        season_windows = windows.loc[windows["season"].eq(season)]
        preseason_window = season_windows.loc[
            season_windows["prediction_type"].eq("preseason")
        ].iloc[0]
        _append_snapshot(
            records,
            window=preseason_window,
            season_teams=season_teams,
            history=history,
            tracker=tracker,
            season_start_ratings=season_start_ratings,
            config=config,
            target_lookup=target_lookup,
            target_seasons=target_seasons,
        )

        season_matches = matches.loc[matches["season"].eq(season)]
        regular = season_matches.loc[season_matches["stage"].eq("regular_season")]
        for week in sorted(int(value) for value in regular["week"].dropna().unique()):
            week_matches = regular.loc[regular["week"].eq(week)].sort_values(
                ["completion_date", "match_id"]
            )
            for match in week_matches.itertuples(index=False):
                _append_match_history(match, tracker, history)
            weekly_window = season_windows.loc[
                season_windows["prediction_type"].eq("weekly")
                & season_windows["completed_week"].eq(week)
            ].iloc[0]
            _append_snapshot(
                records,
                window=weekly_window,
                season_teams=season_teams,
                history=history,
                tracker=tracker,
                season_start_ratings=season_start_ratings,
                config=config,
                target_lookup=target_lookup,
                target_seasons=target_seasons,
            )

        playoffs = season_matches.loc[season_matches["stage"].eq("playoffs")].sort_values(
            ["completion_date", "match_id"]
        )
        for match in playoffs.itertuples(index=False):
            _append_match_history(match, tracker, history)

    features = pd.DataFrame.from_records(records)
    features, roster_metadata = add_roster_features(features, tables["players"], config["roster"])
    features["season"] = features["season"].astype("Int64")
    features["completed_week"] = features["completed_week"].astype("Int64")
    features["target_available"] = features["target_available"].astype("boolean")
    features["champion"] = features["champion"].astype("boolean")
    integer_features = [
        "prior_season_regular_matches",
        "prior_season_playoff_matches",
        "prior_3_season_matches",
        "current_regular_matches",
        "current_regular_wins",
        "current_regular_losses",
        "current_regular_draws",
        "current_regular_game_differential",
    ]
    for column in integer_features:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("Int64")
    return features.sort_values(["season", "snapshot_id", "team_id"]).reset_index(
        drop=True
    ), roster_metadata


def build_feature_report(
    features: pd.DataFrame,
    windows: pd.DataFrame,
    roster_metadata: dict[str, Any],
) -> dict[str, Any]:
    expected_rows = int(windows["team_count"].sum())
    duplicate_rows = int(features.duplicated(["snapshot_id", "franchise_slot_id"]).sum())
    preseason_nonzero = int(
        features.loc[features["prediction_type"].eq("preseason"), "current_regular_matches"]
        .ne(0)
        .sum()
    )
    date_leaks = int(
        features["latest_match_completion_date_used"]
        .gt(features["feature_cutoff_date"])
        .fillna(False)
        .sum()
    )
    champion_counts = (
        features.loc[features["target_available"]].groupby("snapshot_id")["champion"].sum()
    )
    current_match_totals = features.groupby("snapshot_id", as_index=False)[
        "current_regular_matches"
    ].sum()
    expected_match_totals = windows[["snapshot_id", "available_regular_match_count"]].copy()
    expected_match_totals["expected_team_match_rows"] = (
        expected_match_totals["available_regular_match_count"] * 2
    )
    match_validation = current_match_totals.merge(expected_match_totals, on="snapshot_id")
    match_count_mismatches = int(
        match_validation["current_regular_matches"]
        .ne(match_validation["expected_team_match_rows"])
        .sum()
    )
    checks = [
        {
            "check_id": "expected_feature_rows",
            "status": "pass" if len(features) == expected_rows else "fail",
            "count": abs(len(features) - expected_rows),
        },
        {
            "check_id": "unique_snapshot_team",
            "status": "pass" if duplicate_rows == 0 else "fail",
            "count": duplicate_rows,
        },
        {
            "check_id": "one_champion_per_historical_snapshot",
            "status": "pass" if champion_counts.eq(1).all() else "fail",
            "count": int(champion_counts.ne(1).sum()),
        },
        {
            "check_id": "preseason_has_no_current_results",
            "status": "pass" if preseason_nonzero == 0 else "fail",
            "count": preseason_nonzero,
        },
        {
            "check_id": "latest_source_date_not_after_cutoff",
            "status": "pass" if date_leaks == 0 else "fail",
            "count": date_leaks,
        },
        {
            "check_id": "snapshot_match_counts_reconcile",
            "status": "pass" if match_count_mismatches == 0 else "fail",
            "count": match_count_mismatches,
        },
    ]
    feature_missing = pd.DataFrame(
        [
            {
                "feature": column,
                "missing_count": int(features[column].isna().sum()),
                "missing_pct": round(float(features[column].isna().mean() * 100), 2),
            }
            for column in model_feature_columns()
        ]
    ).sort_values(["missing_pct", "feature"], ascending=[False, True])
    disabled_groups = (
        [] if roster_metadata["current_roster_features_enabled"] else ["current_roster_temporal"]
    )
    enabled_columns = [
        column
        for group, columns in FEATURE_GROUPS.items()
        if group not in disabled_groups
        for column in columns
    ]
    return {
        "report_version": "1.0",
        "feature_version": str(features["feature_version"].iloc[0]),
        "snapshot_count": int(features["snapshot_id"].nunique()),
        "feature_row_count": len(features),
        "feature_column_count": len(model_feature_columns()),
        "enabled_feature_column_count": len(enabled_columns),
        "season_min": int(features["season"].min()),
        "season_max": int(features["season"].max()),
        "blocking_issue_count": sum(check["status"] == "fail" for check in checks),
        "checks": checks,
        "feature_groups": FEATURE_GROUPS,
        "disabled_feature_groups": disabled_groups,
        "enabled_feature_columns": enabled_columns,
        "identifier_columns": IDENTIFIER_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "roster": roster_metadata,
        "feature_missingness": dataframe_records(feature_missing),
    }


def write_snapshot_outputs(
    features: pd.DataFrame,
    report: dict[str, Any],
    feature_path: Path,
    report_path: Path,
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    write_json(report, report_path)
