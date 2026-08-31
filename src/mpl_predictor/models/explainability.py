from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json
from mpl_predictor.models.walk_forward import _numeric

FEATURE_LABELS = {
    "elo_rating_diff": "Selisih Elo",
    "current_matches_diff": "Selisih jumlah match S18",
    "current_match_win_rate_diff": "Selisih win rate match S18",
    "current_game_win_rate_diff": "Selisih win rate game S18",
    "current_game_diff_per_match_diff": "Selisih game differential per match",
    "current_form_3_diff": "Selisih form 3 match",
    "current_form_5_diff": "Selisih form 5 match",
    "current_sos_elo_avg_diff": "Selisih strength of schedule Elo",
    "prior_regular_match_win_rate_diff": "Selisih win rate regular season sebelumnya",
    "prior_regular_game_win_rate_diff": "Selisih win rate game musim sebelumnya",
    "prior_regular_game_diff_per_match_diff": "Selisih game differential musim sebelumnya",
    "prior_3_season_match_win_rate_diff": "Selisih win rate match 3 musim",
    "prior_3_season_game_win_rate_diff": "Selisih win rate game 3 musim",
    "prior_3_season_game_diff_per_match_diff": "Selisih game differential 3 musim",
    "rest_days_diff": "Selisih hari istirahat",
}


def _display_feature_name(transformed_name: str) -> tuple[str, str]:
    prefix = "missingindicator_"
    if transformed_name.startswith(prefix):
        source = transformed_name.removeprefix(prefix)
        return source, f"Indikator data kosong: {FEATURE_LABELS.get(source, source)}"
    return transformed_name, FEATURE_LABELS.get(transformed_name, transformed_name)


def build_global_importance(artifact: dict[str, Any]) -> pd.DataFrame:
    pipeline = artifact["pipeline"]
    columns = list(artifact["feature_columns"])
    names = pipeline.named_steps["imputer"].get_feature_names_out(columns)
    coefficients = pipeline.named_steps["model"].coef_[0]
    records = []
    for transformed_name, coefficient in zip(names, coefficients, strict=True):
        source, label = _display_feature_name(str(transformed_name))
        records.append(
            {
                "transformed_feature": str(transformed_name),
                "source_feature": source,
                "feature_label": label,
                "coefficient": float(coefficient),
                "absolute_importance": abs(float(coefficient)),
                "effect_direction_for_positive_difference": (
                    "team_a" if coefficient > 0 else "team_b"
                ),
            }
        )
    result = pd.DataFrame.from_records(records).sort_values(
        ["absolute_importance", "transformed_feature"], ascending=[False, True]
    )
    result["importance_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def build_match_explanations(
    artifact: dict[str, Any], match_probabilities: pd.DataFrame
) -> pd.DataFrame:
    """Return forward raw-logit contributions for the latest unplayed matches."""
    latest_order = match_probabilities["snapshot_order"].max()
    matches = match_probabilities.loc[
        match_probabilities["snapshot_order"].eq(latest_order)
        & match_probabilities["status"].eq("scheduled")
    ].copy()
    columns = list(artifact["feature_columns"])
    pipeline = artifact["pipeline"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    names = imputer.get_feature_names_out(columns)
    numeric = _numeric(matches, columns)
    imputed = imputer.transform(numeric)
    standardized = scaler.transform(imputed)
    contributions = standardized * model.coef_[0]
    raw_logits = model.intercept_[0] + contributions.sum(axis=1)
    records = []
    for row_position, row in enumerate(matches.itertuples(index=False)):
        for feature_position, transformed_name in enumerate(names):
            source, label = _display_feature_name(str(transformed_name))
            contribution = float(contributions[row_position, feature_position])
            if str(transformed_name).startswith("missingindicator_"):
                source_index = columns.index(source)
                raw_value = int(pd.isna(numeric.iloc[row_position, source_index]))
            else:
                raw_value = numeric.iloc[row_position][source]
            records.append(
                {
                    "snapshot_id": str(row.snapshot_id),
                    "match_id": str(row.match_id),
                    "week": int(row.week),
                    "scheduled_at": row.scheduled_at,
                    "team_a_id": str(row.team_a_id),
                    "team_b_id": str(row.team_b_id),
                    "team_a_win_probability": float(row.team_a_win_probability),
                    "raw_forward_logit": float(raw_logits[row_position]),
                    "transformed_feature": str(transformed_name),
                    "source_feature": source,
                    "feature_label": label,
                    "raw_value": None if pd.isna(raw_value) else float(raw_value),
                    "standardized_value": float(standardized[row_position, feature_position]),
                    "coefficient": float(model.coef_[0][feature_position]),
                    "contribution": contribution,
                    "absolute_contribution": abs(contribution),
                    "favors_team_id": str(row.team_a_id if contribution >= 0 else row.team_b_id),
                    "explanation_scope": "forward_raw_logit_before_symmetry_and_calibration",
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["contribution_rank"] = result.groupby("match_id")["absolute_contribution"].rank(
        method="first", ascending=False
    )
    return result.sort_values(["scheduled_at", "match_id", "contribution_rank"]).reset_index(
        drop=True
    )


def build_team_explanations(
    predictions: pd.DataFrame, match_probabilities: pd.DataFrame
) -> pd.DataFrame:
    preseason = predictions.loc[predictions["prediction_type"].eq("preseason")].set_index("team_id")
    latest_order = predictions["snapshot_order"].max()
    latest = predictions.loc[predictions["snapshot_order"].eq(latest_order)].set_index("team_id")
    latest_matches = match_probabilities.loc[
        match_probabilities["snapshot_order"].eq(latest_order)
        & match_probabilities["status"].eq("scheduled")
    ]
    schedule_strength: dict[str, list[float]] = {}
    for row in latest_matches.itertuples(index=False):
        schedule_strength.setdefault(str(row.team_a_id), []).append(
            float(row.team_a_win_probability)
        )
        schedule_strength.setdefault(str(row.team_b_id), []).append(
            1.0 - float(row.team_a_win_probability)
        )
    records = []
    for team_id, row in latest.iterrows():
        preseason_probability = float(preseason.loc[team_id, "champion_probability"])
        current_probability = float(row["champion_probability"])
        change = current_probability - preseason_probability
        records.append(
            {
                "snapshot_id": str(row["snapshot_id"]),
                "team_id": str(team_id),
                "team_name": str(row["team_name"]),
                "current_rank": int(row["current_rank"]),
                "current_match_wins": int(row["current_match_wins"]),
                "current_match_losses": int(row["current_match_losses"]),
                "expected_regular_rank": float(row["expected_regular_rank"]),
                "preseason_champion_probability": preseason_probability,
                "current_champion_probability": current_probability,
                "champion_probability_change": change,
                "playoff_probability": float(row["playoff_probability"]),
                "grand_final_probability": float(row["grand_final_probability"]),
                "mean_remaining_match_win_probability": float(
                    np.mean(schedule_strength.get(str(team_id), [0.5]))
                ),
                "change_direction": "naik" if change > 0 else "turun" if change < 0 else "tetap",
            }
        )
    result = pd.DataFrame.from_records(records).sort_values(
        ["current_champion_probability", "team_id"], ascending=[False, True]
    )
    result["current_champion_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def build_explainability_report(
    artifact: dict[str, Any],
    global_importance: pd.DataFrame,
    match_explanations: pd.DataFrame,
    team_explanations: pd.DataFrame,
) -> dict[str, Any]:
    top_global = global_importance.head(10)
    top_local = (
        match_explanations.loc[match_explanations["contribution_rank"].le(3)]
        if not match_explanations.empty
        else match_explanations
    )
    return {
        "report_version": "1.0",
        "season": 18,
        "model_name": artifact["model_name"],
        "methods": {
            "global": "Absolute standardized logistic coefficient.",
            "match_local": (
                "Feature contribution pada raw forward logit sebelum side-symmetry dan "
                "kalibrasi Platt."
            ),
            "champion_team": (
                "Perubahan probabilitas Monte Carlo dari pramusim ke snapshot terbaru; "
                "bukan dekomposisi aditif atau causal attribution."
            ),
        },
        "scope": {
            "global_transformed_feature_count": len(global_importance),
            "explained_upcoming_match_count": int(match_explanations["match_id"].nunique())
            if not match_explanations.empty
            else 0,
            "local_contribution_row_count": len(match_explanations),
            "team_count": len(team_explanations),
        },
        "interpretation_guards": [
            "Kontribusi positif mendukung team A; kontribusi negatif mendukung team B.",
            "Kontribusi local menjelaskan raw logit, bukan keseluruhan proses Monte Carlo.",
            "Korelasi fitur tidak boleh ditafsirkan sebagai sebab-akibat.",
        ],
        "top_global_features": dataframe_records(top_global),
        "top_match_contributions": dataframe_records(top_local),
        "team_probability_changes": dataframe_records(team_explanations),
    }


def write_explainability_outputs(
    global_importance: pd.DataFrame,
    match_explanations: pd.DataFrame,
    team_explanations: pd.DataFrame,
    report: dict[str, Any],
    prediction_dir: Path,
    report_path: Path,
) -> dict[str, Path]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "global": prediction_dir / "season18_global_feature_importance.parquet",
        "matches": prediction_dir / "season18_match_explanations.parquet",
        "teams": prediction_dir / "season18_team_explanations.parquet",
    }
    global_importance.to_parquet(outputs["global"], index=False)
    match_explanations.to_parquet(outputs["matches"], index=False)
    team_explanations.to_parquet(outputs["teams"], index=False)
    write_json(report, report_path)
    return outputs
