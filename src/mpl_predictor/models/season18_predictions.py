from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.analysis.common import write_json
from mpl_predictor.models.simulation import (
    build_season18_match_probabilities,
    simulate_season18,
)


def load_season18_tables(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = pd.read_csv(directory / "teams.csv")
    rosters = pd.read_csv(directory / "rosters.csv")
    schedule = pd.read_csv(directory / "schedule_results.csv")
    schedule["scheduled_at"] = pd.to_datetime(schedule["scheduled_at"], utc=True).dt.tz_convert(
        "Asia/Jakarta"
    )
    for column in ("team_a_score", "team_b_score"):
        schedule[column] = pd.to_numeric(schedule[column], errors="coerce").astype("Int64")
    for column in ("valid_from", "valid_to", "observed_at"):
        if column in rosters:
            rosters[column] = pd.to_datetime(rosters[column], errors="coerce")
    return teams, rosters, schedule


def build_season18_prediction_windows(schedule: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-safe preseason window and every fully completed weekly window."""
    first_match = schedule["scheduled_at"].min()
    records = [
        {
            "snapshot_id": "S18_PRE",
            "prediction_type": "preseason",
            "completed_week": pd.NA,
            "feature_cutoff_date": (first_match - pd.Timedelta(days=1)).date(),
            "available_result_count": 0,
        }
    ]
    for week in sorted(int(value) for value in schedule["week"].unique()):
        week_rows = schedule.loc[schedule["week"].eq(week)]
        if week_rows.empty or not week_rows["status"].eq("completed").all():
            break
        records.append(
            {
                "snapshot_id": f"S18_W{week:02}",
                "prediction_type": "weekly",
                "completed_week": week,
                "feature_cutoff_date": week_rows["scheduled_at"].max().date(),
                "available_result_count": int(schedule["week"].le(week).sum()),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["completed_week"] = result["completed_week"].astype("Int64")
    result["available_result_count"] = result["available_result_count"].astype("Int64")
    return result


def schedule_as_of_window(schedule: pd.DataFrame, window: pd.Series) -> pd.DataFrame:
    """Hide every result that was unavailable at the requested S18 snapshot."""
    result = schedule.copy()
    completed_week = window["completed_week"]
    available = (
        result["status"].eq("completed") & result["week"].le(int(completed_week))
        if pd.notna(completed_week)
        else pd.Series(False, index=result.index)
    )
    hidden = ~available
    for column in ("team_a_score", "team_b_score", "winner_team_id", "winner_side"):
        result.loc[hidden, column] = pd.NA
    result.loc[hidden, "status"] = "scheduled"
    result["observed_at"] = str(window["feature_cutoff_date"])
    result["snapshot_id"] = str(window["snapshot_id"])
    result["prediction_type"] = str(window["prediction_type"])
    result["completed_week"] = completed_week
    return result


def build_season18_prediction_history(
    tables: dict[str, pd.DataFrame],
    teams: pd.DataFrame,
    rosters: pd.DataFrame,
    schedule: pd.DataFrame,
    feature_config: dict[str, Any],
    artifact: dict[str, Any],
    simulation_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Reconstruct preseason and create every currently available weekly prediction."""
    windows = build_season18_prediction_windows(schedule)
    prediction_frames = []
    match_frames = []
    snapshot_summaries = []
    latest_report: dict[str, Any] = {}

    for window in windows.to_dict(orient="records"):
        window_series = pd.Series(window)
        snapshot_schedule = schedule_as_of_window(schedule, window_series)
        probabilities, tracker, history = build_season18_match_probabilities(
            tables, snapshot_schedule, teams, feature_config, artifact
        )
        predictions, simulation_report = simulate_season18(
            tables,
            snapshot_schedule,
            teams,
            probabilities,
            tracker,
            history,
            artifact,
            simulation_config,
        )
        cutoff = pd.Timestamp(window["feature_cutoff_date"])
        roster_available = rosters.loc[
            rosters["member_type"].eq("player")
            & rosters["valid_from"].notna()
            & rosters["valid_from"].le(cutoff)
            & (rosters["valid_to"].isna() | rosters["valid_to"].ge(cutoff))
        ]
        metadata = {
            "snapshot_id": window["snapshot_id"],
            "prediction_type": window["prediction_type"],
            "completed_week": window["completed_week"],
            "feature_cutoff_date": str(window["feature_cutoff_date"]),
            "available_result_count": int(window["available_result_count"]),
            "roster_player_rows_available_at_cutoff": len(roster_available),
            "roster_feature_rows_used": 0,
        }
        for key, value in metadata.items():
            predictions[key] = value
            probabilities[key] = value
        prediction_frames.append(predictions)
        match_frames.append(probabilities)
        leader = predictions.sort_values("champion_probability", ascending=False).iloc[0]
        snapshot_summaries.append(
            {
                **metadata,
                "completed_match_count": simulation_report["input_scope"]["completed_match_count"],
                "simulated_regular_match_count": simulation_report["input_scope"][
                    "simulated_regular_match_count"
                ],
                "leader_team_id": str(leader["team_id"]),
                "leader_champion_probability": float(leader["champion_probability"]),
                "probability_sum": float(predictions["champion_probability"].sum()),
            }
        )
        latest_report = simulation_report

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_matches = pd.concat(match_frames, ignore_index=True)
    snapshot_order = {
        snapshot_id: index for index, snapshot_id in enumerate(windows["snapshot_id"])
    }
    all_predictions["snapshot_order"] = all_predictions["snapshot_id"].map(snapshot_order)
    all_matches["snapshot_order"] = all_matches["snapshot_id"].map(snapshot_order)
    report = {
        "report_version": "1.0",
        "season": 18,
        "method": "retrospective_as_of_reconstruction_and_weekly_update",
        "source_data_observed_at": str(schedule["observed_at"].iloc[0]),
        "snapshot_count": len(windows),
        "preseason_snapshot_count": int(windows["prediction_type"].eq("preseason").sum()),
        "weekly_snapshot_count": int(windows["prediction_type"].eq("weekly").sum()),
        "latest_completed_week": (
            int(windows["completed_week"].max())
            if windows["completed_week"].notna().any()
            else None
        ),
        "leakage_guards": {
            "preseason_results_used": 0,
            "future_week_results_hidden": True,
            "roster_rule": "valid_from <= feature_cutoff_date",
            "roster_model_usage": (
                "Tidak digunakan model match final v1 karena training historis tidak memiliki "
                "roster bertanggal yang sebanding."
            ),
            "reconstruction_note": (
                "Daftar peserta dan jadwal berasal dari snapshot resmi yang dikumpulkan "
                "setelah musim dimulai; seluruh outcome S18 tetap disembunyikan sesuai cutoff."
            ),
        },
        "validation": {
            "invalid_probability_sum_count": sum(
                abs(item["probability_sum"] - 1.0) > 1e-9 for item in snapshot_summaries
            ),
            "preseason_available_result_count": snapshot_summaries[0]["available_result_count"],
        },
        "snapshots": snapshot_summaries,
    }
    return all_predictions, all_matches, report, latest_report


def write_season18_prediction_history(
    predictions: pd.DataFrame,
    match_probabilities: pd.DataFrame,
    report: dict[str, Any],
    latest_report: dict[str, Any],
    prediction_dir: Path,
    report_path: Path,
    latest_report_path: Path,
) -> dict[str, Path]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "history": prediction_dir / "season18_snapshot_predictions.parquet",
        "preseason": prediction_dir / "season18_preseason_prediction.parquet",
        "weekly": prediction_dir / "season18_weekly_predictions.parquet",
        "matches": prediction_dir / "season18_snapshot_match_probabilities.parquet",
        "latest": prediction_dir / "season18_simulation.parquet",
        "latest_matches": prediction_dir / "season18_match_probabilities.parquet",
    }
    predictions.to_parquet(outputs["history"], index=False)
    predictions.loc[predictions["prediction_type"].eq("preseason")].to_parquet(
        outputs["preseason"], index=False
    )
    predictions.loc[predictions["prediction_type"].eq("weekly")].to_parquet(
        outputs["weekly"], index=False
    )
    match_probabilities.to_parquet(outputs["matches"], index=False)
    latest_snapshot = predictions["snapshot_order"].max()
    predictions.loc[predictions["snapshot_order"].eq(latest_snapshot)].to_parquet(
        outputs["latest"], index=False
    )
    match_probabilities.loc[match_probabilities["snapshot_order"].eq(latest_snapshot)].to_parquet(
        outputs["latest_matches"], index=False
    )
    write_json(report, report_path)
    write_json(latest_report, latest_report_path)
    return outputs
