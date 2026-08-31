import json
from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json


def load_prediction_policy(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    required = {
        "policy_version",
        "prediction_target",
        "modeling_scope",
        "preseason",
        "weekly",
        "backtesting",
        "current_data_guards",
        "season_18_delivery",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Prediction policy is missing sections: {', '.join(missing)}")
    return policy


def _match_completion_dates(matches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    game_dates = (
        games.groupby(["season", "match_id"], as_index=False)["date"]
        .max()
        .rename(columns={"date": "last_game_date"})
    )
    result = matches.merge(game_dates, on=["season", "match_id"], how="left")
    result["completion_date"] = result[["date", "last_game_date"]].max(axis=1)
    return result


def build_prediction_windows(
    tables: dict[str, pd.DataFrame], policy: dict[str, Any]
) -> pd.DataFrame:
    """Create leakage-safe preseason and weekly historical snapshot cutoffs."""
    matches = _match_completion_dates(tables["matches"], tables["games"])
    primary_start, primary_end = policy["modeling_scope"]["primary_seasons"]
    regular = matches.loc[
        matches["season"].between(primary_start, primary_end)
        & matches["stage"].eq("regular_season")
        & matches["week"].notna()
    ].copy()
    teams = tables["teams"]
    games = tables["games"]
    records: list[dict[str, Any]] = []

    for season in sorted(int(value) for value in regular["season"].unique()):
        season_matches = regular.loc[regular["season"].eq(season)].copy()
        first_match_date = season_matches["date"].min()
        last_completion_date = season_matches["completion_date"].max()
        team_count = int(teams.loc[teams["season"].eq(season), "team_id"].nunique())
        records.append(
            {
                "snapshot_id": f"S{season:02}_PRE",
                "season": season,
                "prediction_type": "preseason",
                "completed_week": pd.NA,
                "feature_cutoff_date": first_match_date - pd.Timedelta(days=1),
                "first_regular_season_date": first_match_date,
                "last_regular_season_completion_date": last_completion_date,
                "team_count": team_count,
                "available_regular_match_count": 0,
                "available_game_count": 0,
                "available_game_outcome_count": 0,
            }
        )

        weeks = sorted(int(value) for value in season_matches["week"].unique())
        for week in weeks:
            available_matches = season_matches.loc[season_matches["week"].le(week)]
            cutoff = available_matches["completion_date"].max()
            match_keys = available_matches[["season", "match_id"]].drop_duplicates()
            available_games = games.merge(match_keys, on=["season", "match_id"], how="inner")
            records.append(
                {
                    "snapshot_id": f"S{season:02}_W{week:02}",
                    "season": season,
                    "prediction_type": "weekly",
                    "completed_week": week,
                    "feature_cutoff_date": cutoff,
                    "first_regular_season_date": first_match_date,
                    "last_regular_season_completion_date": last_completion_date,
                    "team_count": team_count,
                    "available_regular_match_count": len(available_matches),
                    "available_game_count": len(available_games),
                    "available_game_outcome_count": int(
                        available_games["winner_side"].notna().sum()
                    ),
                }
            )

    result = pd.DataFrame.from_records(records)
    result["season"] = result["season"].astype("Int64")
    result["completed_week"] = result["completed_week"].astype("Int64")
    for column in (
        "team_count",
        "available_regular_match_count",
        "available_game_count",
        "available_game_outcome_count",
    ):
        result[column] = result[column].astype("Int64")
    return result


def build_prediction_policy_report(policy: dict[str, Any], windows: pd.DataFrame) -> dict[str, Any]:
    per_season = (
        windows.groupby("season", as_index=False)
        .agg(
            preseason_snapshot_count=(
                "prediction_type",
                lambda values: int(values.eq("preseason").sum()),
            ),
            weekly_snapshot_count=(
                "prediction_type",
                lambda values: int(values.eq("weekly").sum()),
            ),
            first_cutoff_date=("feature_cutoff_date", "min"),
            last_cutoff_date=("feature_cutoff_date", "max"),
        )
        .sort_values("season")
    )
    return {
        "report_version": "1.0",
        "policy": policy,
        "historical_windows": {
            "season_min": int(windows["season"].min()),
            "season_max": int(windows["season"].max()),
            "snapshot_count": len(windows),
            "preseason_snapshot_count": int(windows["prediction_type"].eq("preseason").sum()),
            "weekly_snapshot_count": int(windows["prediction_type"].eq("weekly").sum()),
            "by_season": dataframe_records(per_season),
        },
        "season_18_status": {
            "windows_generated": False,
            "reason": "Jadwal dan hasil Season 18 belum berada pada canonical dataset.",
            "next_action": (
                "Tambahkan input Season 18, canonicalize, lalu buat snapshot dengan policy "
                "yang sama."
            ),
        },
    }


def write_prediction_outputs(
    report: dict[str, Any],
    windows: pd.DataFrame,
    report_path: Path,
    windows_path: Path,
) -> None:
    write_json(report, report_path)
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    output = windows.copy()
    for column in (
        "feature_cutoff_date",
        "first_regular_season_date",
        "last_regular_season_completion_date",
    ):
        output[column] = output[column].dt.strftime("%Y-%m-%d")
    output.to_csv(windows_path, index=False)
