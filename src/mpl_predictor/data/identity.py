import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def player_key(nickname: Any) -> str | None:
    """Return a conservative key for formatting-only nickname aliases."""
    if nickname is None or pd.isna(nickname):
        return None
    normalized = unicodedata.normalize("NFKC", str(nickname)).casefold().strip()
    key = "".join(character for character in normalized if character.isalnum())
    return key or None


def _player_id(key: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "", key)[:16].upper() or "PLAYER"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8].upper()
    return f"PLY_{readable}_{digest}"


def load_team_identity_rules(path: Path) -> pd.DataFrame:
    rules = pd.read_csv(path, dtype=str, keep_default_na=False)
    rules["start_season"] = pd.to_numeric(rules["start_season"], errors="raise").astype(int)
    rules["end_season"] = pd.to_numeric(rules["end_season"], errors="raise").astype(int)
    for column in (
        "source_team_id",
        "organization_id",
        "organization_name",
        "franchise_slot_id",
        "mapping_basis",
        "evidence_url",
    ):
        rules[column] = rules[column].str.strip().replace("", pd.NA).astype("string")
    return rules


def load_player_alias_overrides(path: Path) -> dict[str, str]:
    overrides = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_columns = {"alias_key", "canonical_key", "reason"}
    if not required_columns.issubset(overrides.columns):
        missing = sorted(required_columns - set(overrides.columns))
        raise ValueError(f"Player alias overrides are missing columns: {', '.join(missing)}")

    mapping: dict[str, str] = {}
    for row in overrides.itertuples():
        alias = player_key(row.alias_key)
        canonical = player_key(row.canonical_key)
        if alias is None or canonical is None:
            raise ValueError("Player alias keys cannot be empty.")
        if alias in mapping:
            raise ValueError(f"Duplicate player alias override: {alias}")
        mapping[alias] = canonical

    for alias, canonical in mapping.items():
        if canonical in mapping:
            raise ValueError(
                f"Player alias override must point directly to a final key: {alias} -> {canonical}"
            )
    return mapping


def build_team_identity(teams: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Map every season team to an organization and, in S4+, a franchise slot."""
    records: list[dict[str, Any]] = []
    for row in teams.itertuples():
        candidates = rules.loc[
            rules["source_team_id"].eq(row.team_id)
            & rules["start_season"].le(int(row.season))
            & rules["end_season"].ge(int(row.season))
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one identity rule for S{int(row.season):02}/{row.team_id}, "
                f"found {len(candidates)}."
            )
        rule = candidates.iloc[0]
        record = row._asdict()
        record.update(
            {
                "organization_id": rule["organization_id"],
                "organization_name": rule["organization_name"],
                "franchise_slot_id": rule["franchise_slot_id"],
                "mapping_basis": rule["mapping_basis"],
                "mapping_evidence_url": rule["evidence_url"],
                "identity_review_required": False,
            }
        )
        records.append(record)

    result = pd.DataFrame.from_records(records)
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column].dtype):
            result[column] = result[column].astype("string")
    result["identity_review_required"] = result["identity_review_required"].astype("boolean")
    return result


def _canonical_nickname(group: pd.DataFrame) -> str:
    counts = group["player"].value_counts()
    most_common = set(counts.loc[counts.eq(counts.max())].index)
    candidates = group.loc[group["player"].isin(most_common)].sort_values(
        ["season", "player"], ascending=[False, True]
    )
    return str(candidates.iloc[0]["player"])


def build_player_identity(
    players: pd.DataFrame,
    player_stats: pd.DataFrame,
    team_identity: pd.DataFrame,
    alias_overrides: dict[str, str],
) -> pd.DataFrame:
    """Build deterministic player IDs and flag short or ambiguous nickname keys for review."""
    roster = players[["season", "team_id", "player", "role"]].assign(source_table="players")
    stats = player_stats[["season", "team_id", "player", "role"]].assign(
        source_table="player_season_stats"
    )
    observations = pd.concat([roster, stats], ignore_index=True)
    observations["player_key_raw"] = observations["player"].map(player_key).astype("string")
    observations["player_key"] = (
        observations["player_key_raw"]
        .map(lambda key: alias_overrides.get(str(key), key) if pd.notna(key) else None)
        .astype("string")
    )

    organization_lookup = {
        (int(row.season), str(row.team_id)): str(row.organization_id)
        for row in team_identity.itertuples()
    }
    observations["organization_id"] = pd.Series(
        [
            organization_lookup.get((int(season), str(team_id)))
            for season, team_id in zip(observations["season"], observations["team_id"], strict=True)
        ],
        dtype="string",
    )

    records: list[dict[str, Any]] = []
    for key, group in observations.groupby("player_key", dropna=False):
        if pd.isna(key):
            raise ValueError("A player nickname produced an empty identity key.")
        key = str(key)
        same_season_team_counts = group.groupby("season")["team_id"].nunique()
        ambiguous_same_season = bool(same_season_team_counts.gt(1).any())
        short_key = len(key) <= 2
        aliases = sorted(str(value) for value in group["player"].dropna().unique())
        team_ids = sorted(str(value) for value in group["team_id"].dropna().unique())
        organization_ids = sorted(
            str(value) for value in group["organization_id"].dropna().unique()
        )
        identity_confidence = "low" if len(key) == 1 else "medium" if short_key else "high"
        records.append(
            {
                "player_id": _player_id(key),
                "player_key": key,
                "canonical_nickname": _canonical_nickname(group),
                "aliases": json.dumps(aliases, ensure_ascii=False),
                "first_season": int(group["season"].min()),
                "last_season": int(group["season"].max()),
                "team_ids": json.dumps(team_ids, ensure_ascii=False),
                "organization_ids": json.dumps(organization_ids, ensure_ascii=False),
                "observation_count": len(group),
                "identity_confidence": identity_confidence,
                "identity_review_required": short_key or ambiguous_same_season,
                "ambiguous_same_season": ambiguous_same_season,
            }
        )

    result = pd.DataFrame.from_records(records).sort_values("player_id").reset_index(drop=True)
    for column in (
        "player_id",
        "player_key",
        "canonical_nickname",
        "aliases",
        "team_ids",
        "organization_ids",
        "identity_confidence",
    ):
        result[column] = result[column].astype("string")
    for column in ("first_season", "last_season", "observation_count"):
        result[column] = result[column].astype("Int64")
    for column in ("identity_review_required", "ambiguous_same_season"):
        result[column] = result[column].astype("boolean")
    return result


def _team_identity_lookup(team_identity: pd.DataFrame, field: str) -> dict[tuple[int, str], str]:
    return {
        (int(row.season), str(row.team_id)): str(getattr(row, field))
        for row in team_identity.itertuples()
        if pd.notna(getattr(row, field))
    }


def _enrich_team_reference(
    frame: pd.DataFrame,
    team_id_column: str,
    prefix: str,
    team_identity: pd.DataFrame,
) -> None:
    for field in ("organization_id", "franchise_slot_id"):
        lookup = _team_identity_lookup(team_identity, field)
        frame[f"{prefix}{field}"] = pd.Series(
            [
                lookup.get((int(season), str(team_id))) if pd.notna(team_id) else None
                for season, team_id in zip(frame["season"], frame[team_id_column], strict=True)
            ],
            dtype="string",
        )


def _enrich_player_reference(
    frame: pd.DataFrame,
    player_identity: pd.DataFrame,
    alias_overrides: dict[str, str],
) -> None:
    lookup = player_identity.set_index("player_key")
    frame["player_key_raw"] = frame["player"].map(player_key).astype("string")
    frame["player_key"] = (
        frame["player_key_raw"]
        .map(lambda key: alias_overrides.get(str(key), key) if pd.notna(key) else None)
        .astype("string")
    )
    frame["player_id"] = frame["player_key"].map(lookup["player_id"]).astype("string")
    frame["canonical_nickname"] = (
        frame["player_key"].map(lookup["canonical_nickname"]).astype("string")
    )
    frame["player_identity_confidence"] = (
        frame["player_key"].map(lookup["identity_confidence"]).astype("string")
    )
    frame["player_identity_review_required"] = (
        frame["player_key"].map(lookup["identity_review_required"]).astype("boolean")
    )


def build_canonical_tables(
    normalized_tables: dict[str, pd.DataFrame],
    rules: pd.DataFrame,
    alias_overrides: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Enrich normalized tables with team, franchise-slot, and player identities."""
    team_identity = build_team_identity(normalized_tables["teams"], rules)
    player_identity = build_player_identity(
        normalized_tables["players"],
        normalized_tables["player_season_stats"],
        team_identity,
        alias_overrides,
    )

    canonical = {
        table: frame.copy() for table, frame in normalized_tables.items() if table != "teams"
    }
    canonical["teams"] = team_identity
    canonical["player_identity"] = player_identity

    for table in ("matches", "games"):
        frame = canonical[table]
        _enrich_team_reference(frame, "team_a_id", "team_a_", team_identity)
        _enrich_team_reference(frame, "team_b_id", "team_b_", team_identity)
        _enrich_team_reference(frame, "winner_team_id", "winner_", team_identity)

    for table in ("players", "championships", "drafts", "player_season_stats"):
        _enrich_team_reference(canonical[table], "team_id", "", team_identity)

    for table in ("players", "player_season_stats"):
        _enrich_player_reference(canonical[table], player_identity, alias_overrides)

    return canonical


def identity_summary(canonical_tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    teams = canonical_tables["teams"]
    players = canonical_tables["player_identity"]
    return {
        "team_season_rows": len(teams),
        "mapped_team_season_rows": int(teams["organization_id"].notna().sum()),
        "organization_count": int(teams["organization_id"].nunique()),
        "franchise_slot_count": int(teams["franchise_slot_id"].nunique()),
        "franchise_rows_without_slot": int(
            teams.loc[teams["season"].ge(4), "franchise_slot_id"].isna().sum()
        ),
        "pre_franchise_rows_with_slot": int(
            teams.loc[teams["season"].lt(4), "franchise_slot_id"].notna().sum()
        ),
        "player_identity_count": len(players),
        "player_alias_group_count": int(players["aliases"].str.contains('", "').sum()),
        "player_review_required_count": int(players["identity_review_required"].sum()),
        "ambiguous_same_season_player_count": int(players["ambiguous_same_season"].sum()),
        "unmapped_roster_player_count": int(canonical_tables["players"]["player_id"].isna().sum()),
        "unmapped_player_stat_count": int(
            canonical_tables["player_season_stats"]["player_id"].isna().sum()
        ),
    }


def write_canonical_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for table, frame in sorted(tables.items()):
        path = output_dir / f"{table}.parquet"
        frame.to_parquet(path, index=False)
        outputs[table] = path
    return outputs


def write_identity_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
