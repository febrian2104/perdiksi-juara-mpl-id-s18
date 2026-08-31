from typing import Any

import pandas as pd


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in columns), None)


def _percentage(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def add_roster_features(
    features: pd.DataFrame,
    players: pd.DataFrame,
    roster_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add safe lagged roster features and optional timestamped current-roster features."""
    result = features.copy()
    working = players.copy()
    excluded_review_rows = 0
    if roster_config["exclude_identity_review_required"]:
        review_mask = working["player_identity_review_required"].fillna(True)
        excluded_review_rows = int(review_mask.sum())
        working = working.loc[~review_mask].copy()

    valid = working.loc[working["franchise_slot_id"].notna() & working["player_id"].notna()].copy()
    roster_groups = {
        (int(season), str(slot)): group.copy()
        for (season, slot), group in valid.groupby(["season", "franchise_slot_id"])
    }
    roster_sets = {key: set(group["player_id"].astype(str)) for key, group in roster_groups.items()}
    player_seasons = {
        str(player_id): set(int(value) for value in group["season"].unique())
        for player_id, group in working.loc[working["player_id"].notna()].groupby("player_id")
    }

    lagged_records = []
    entity_keys = result[["season", "franchise_slot_id"]].drop_duplicates()
    for row in entity_keys.itertuples(index=False):
        season = int(row.season)
        slot = str(row.franchise_slot_id)
        prior_key = (season - 1, slot)
        prior_group = roster_groups.get(prior_key)
        prior_roster = roster_sets.get(prior_key)
        two_seasons_back = roster_sets.get((season - 2, slot))
        record: dict[str, Any] = {
            "season": season,
            "franchise_slot_id": slot,
            "lagged_roster_available": prior_roster is not None,
            "lagged_roster_size": pd.NA,
            "lagged_roster_role_coverage": pd.NA,
            "lagged_roster_returning_count": pd.NA,
            "lagged_roster_continuity": pd.NA,
            "lagged_roster_avg_experience_seasons": pd.NA,
        }
        if prior_roster is not None and prior_group is not None:
            experiences = [
                sum(value <= season - 1 for value in player_seasons[player_id])
                for player_id in prior_roster
            ]
            record["lagged_roster_size"] = len(prior_roster)
            record["lagged_roster_role_coverage"] = _percentage(
                int(prior_group["role"].notna().sum()), len(prior_group)
            )
            record["lagged_roster_avg_experience_seasons"] = round(
                sum(experiences) / len(experiences), 6
            )
            if two_seasons_back is not None:
                retained = prior_roster & two_seasons_back
                record["lagged_roster_returning_count"] = len(retained)
                record["lagged_roster_continuity"] = _percentage(
                    len(retained), len(prior_roster | two_seasons_back)
                )
        lagged_records.append(record)

    lagged = pd.DataFrame.from_records(lagged_records)
    result = result.merge(
        lagged,
        on=["season", "franchise_slot_id"],
        how="left",
        validate="many_to_one",
    )

    start_column = _first_existing(list(players.columns), list(roster_config["start_date_columns"]))
    end_column = _first_existing(list(players.columns), list(roster_config["end_date_columns"]))
    current_columns = {
        "current_roster_size_asof": pd.NA,
        "current_roster_role_coverage_asof": pd.NA,
        "current_roster_retained_count_asof": pd.NA,
        "current_roster_retained_share_asof": pd.NA,
        "current_roster_avg_experience_seasons_asof": pd.NA,
    }

    if start_column is None:
        result["current_roster_temporal_available"] = False
        for column, default in current_columns.items():
            result[column] = default
    else:
        temporal = working.copy()
        temporal[start_column] = pd.to_datetime(temporal[start_column], errors="coerce")
        if end_column is not None:
            temporal[end_column] = pd.to_datetime(temporal[end_column], errors="coerce")
        temporal_records = []
        for row in result[
            ["snapshot_id", "season", "franchise_slot_id", "feature_cutoff_date"]
        ].itertuples(index=False):
            season = int(row.season)
            slot = str(row.franchise_slot_id)
            candidates = temporal.loc[
                temporal["season"].eq(season)
                & temporal["franchise_slot_id"].eq(slot)
                & temporal[start_column].notna()
            ]
            record = {
                "snapshot_id": row.snapshot_id,
                "franchise_slot_id": slot,
                "current_roster_temporal_available": not candidates.empty,
                **current_columns,
            }
            if not candidates.empty:
                active_mask = candidates[start_column].le(row.feature_cutoff_date)
                if end_column is not None:
                    active_mask &= candidates[end_column].isna() | candidates[end_column].ge(
                        row.feature_cutoff_date
                    )
                active = candidates.loc[active_mask]
                active_players = set(active["player_id"].dropna().astype(str))
                prior_roster = roster_sets.get((season - 1, slot), set())
                retained = active_players & prior_roster
                experiences = [
                    sum(value < season for value in player_seasons.get(player_id, set()))
                    for player_id in active_players
                ]
                record.update(
                    {
                        "current_roster_size_asof": len(active_players),
                        "current_roster_role_coverage_asof": _percentage(
                            int(active["role"].notna().sum()), len(active)
                        ),
                        "current_roster_retained_count_asof": len(retained),
                        "current_roster_retained_share_asof": _percentage(
                            len(retained), len(active_players)
                        ),
                        "current_roster_avg_experience_seasons_asof": (
                            round(sum(experiences) / len(experiences), 6) if experiences else None
                        ),
                    }
                )
            temporal_records.append(record)
        temporal_features = pd.DataFrame.from_records(temporal_records)
        result = result.drop(columns=["current_roster_temporal_available"], errors="ignore")
        result = result.merge(
            temporal_features,
            on=["snapshot_id", "franchise_slot_id"],
            how="left",
            validate="one_to_one",
        )

    integer_columns = [
        "lagged_roster_size",
        "lagged_roster_returning_count",
        "current_roster_size_asof",
        "current_roster_retained_count_asof",
    ]
    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    float_columns = [
        "lagged_roster_role_coverage",
        "lagged_roster_continuity",
        "lagged_roster_avg_experience_seasons",
        "current_roster_role_coverage_asof",
        "current_roster_retained_share_asof",
        "current_roster_avg_experience_seasons_asof",
    ]
    for column in float_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Float64")
    for column in (
        "lagged_roster_available",
        "current_roster_temporal_available",
    ):
        result[column] = result[column].fillna(False).astype("boolean")

    metadata = {
        "temporal_start_column": start_column,
        "temporal_end_column": end_column,
        "current_roster_features_enabled": start_column is not None,
        "excluded_identity_review_rows": excluded_review_rows,
        "lagged_roster_entity_count": len(roster_sets),
        "lagged_feature_row_coverage": round(float(result["lagged_roster_available"].mean()), 6),
        "current_temporal_feature_row_coverage": round(
            float(result["current_roster_temporal_available"].mean()), 6
        ),
    }
    return result, metadata
