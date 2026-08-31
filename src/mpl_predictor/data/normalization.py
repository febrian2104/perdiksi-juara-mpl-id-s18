import re
from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.data.catalog import discover_dataset_files

NULL_TOKENS = {"", "-", "--", "n/a", "na", "nan", "none", "null", "—"}
TABLES = (
    "teams",
    "matches",
    "games",
    "players",
    "championships",
    "drafts",
    "player_season_stats",
)

ROLE_ALIASES = {
    "all_role": "all_rounder",
    "all_rounder": "all_rounder",
    "explane": "exp_lane",
    "explaner": "exp_lane",
    "exp_lane": "exp_lane",
    "gold_laner": "gold_lane",
    "goldlane": "gold_lane",
    "goldlaner": "gold_lane",
    "gold_lane": "gold_lane",
    "jungle": "jungler",
    "mid": "mid_lane",
    "midlane": "mid_lane",
    "midlaner": "mid_lane",
    "midlanner": "mid_lane",
    "mid_lane": "mid_lane",
    "multirole": "all_rounder",
    "marskman": "marksman",
    "offlane": "offlaner",
    "offlaner": "offlaner",
    "tanker": "tank",
}

ACTION_ALIASES = {
    "pick_explane": "pick_exp_lane",
    "pick_goldlane": "pick_gold_lane",
    "pick_midlane": "pick_mid_lane",
}

ROUND_ALIASES = {
    "lower_bracket_quarterfinals": "lower_bracket_quarterfinal",
    "lower_bracket_semifinals": "lower_bracket_semifinal",
    "upper_bracket_quarterfinals": "upper_bracket_quarterfinal",
    "upper_bracket_semifinals": "upper_bracket_semifinal",
}


def clean_value(value: Any) -> str | None:
    """Normalize whitespace and common textual missing-value markers."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return None if text.lower() in NULL_TOKENS else text


def snake_case(value: Any) -> str | None:
    """Normalize a categorical value without inventing a replacement for missing data."""
    text = clean_value(value)
    if text is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return normalized or None


def _clean_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object or isinstance(result[column].dtype, pd.StringDtype):
            result[column] = result[column].map(clean_value).astype("string")
    return result


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all recognized CSV files and use the file path as authoritative season metadata."""
    grouped: dict[str, list[pd.DataFrame]] = {table: [] for table in TABLES}
    for dataset_file in discover_dataset_files(data_dir):
        if dataset_file.table not in grouped:
            continue
        frame = pd.read_csv(dataset_file.path, dtype=str, keep_default_na=False)
        frame["season"] = dataset_file.season
        frame["source_file"] = str(dataset_file.path.relative_to(data_dir.resolve()))
        grouped[dataset_file.table].append(frame)

    return {
        table: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for table, frames in grouped.items()
    }


def _team_lookup(teams: pd.DataFrame) -> dict[tuple[int, str], str]:
    return {
        (int(row.season), str(row.team_name)): str(row.team_id)
        for row in teams.itertuples()
        if pd.notna(row.team_name) and pd.notna(row.team_id)
    }


def _add_team_id(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
    lookup: dict[tuple[int, str], str],
) -> None:
    frame[target_column] = pd.Series(
        [
            lookup.get((int(season), str(team))) if pd.notna(team) else None
            for season, team in zip(frame["season"], frame[source_column], strict=True)
        ],
        dtype="string",
    )


def _normalize_teams(frame: pd.DataFrame) -> pd.DataFrame:
    result = _clean_string_columns(frame)
    result["season"] = _to_int(result["season"])
    result["season_team_id"] = pd.Series(
        [
            f"S{int(season):02}:{team_id}"
            for season, team_id in zip(result.season, result.team_id, strict=True)
        ],
        dtype="string",
    )
    return result


def _winner_side_from_scores(score_a: Any, score_b: Any) -> str | None:
    if pd.isna(score_a) or pd.isna(score_b):
        return None
    if int(score_a) == int(score_b):
        return "draw"
    return "team_a" if int(score_a) > int(score_b) else "team_b"


def _winner_side_from_raw(raw: Any, team_a: Any, team_b: Any) -> str | None:
    winner = clean_value(raw)
    if winner in {"team_a", "team_b", "draw"}:
        return winner
    if winner == clean_value(team_a):
        return "team_a"
    if winner == clean_value(team_b):
        return "team_b"
    return None


def _winner_team_id(side: Any, team_a_id: Any, team_b_id: Any) -> str | None:
    if pd.isna(side):
        return None
    if side == "team_a":
        return clean_value(team_a_id)
    if side == "team_b":
        return clean_value(team_b_id)
    return None


def _normalize_matches(frame: pd.DataFrame, lookup: dict[tuple[int, str], str]) -> pd.DataFrame:
    result = _clean_string_columns(frame).rename(columns={"winner": "winner_raw"})
    result["season"] = _to_int(result["season"])
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
    result["week"] = _to_int(result["week"].str.extract(r"(\d+)", expand=False))
    result["team_a_score"] = _to_int(result["team_a_score"])
    result["team_b_score"] = _to_int(result["team_b_score"])
    result["best_of"] = _to_int(result["best_of"])
    result["stage"] = result["stage"].map(snake_case).astype("string")
    normalized_rounds = result["round"].map(snake_case)
    result["round"] = normalized_rounds.map(lambda value: ROUND_ALIASES.get(value, value)).astype(
        "string"
    )
    _add_team_id(result, "team_a", "team_a_id", lookup)
    _add_team_id(result, "team_b", "team_b_id", lookup)
    result["winner_side"] = pd.Series(
        [
            _winner_side_from_scores(score_a, score_b)
            for score_a, score_b in zip(result["team_a_score"], result["team_b_score"], strict=True)
        ],
        dtype="string",
    )
    result["winner_team_id"] = pd.Series(
        [
            _winner_team_id(side, team_a_id, team_b_id)
            for side, team_a_id, team_b_id in zip(
                result["winner_side"], result["team_a_id"], result["team_b_id"], strict=True
            )
        ],
        dtype="string",
    )
    return result


def _normalize_games(frame: pd.DataFrame, lookup: dict[tuple[int, str], str]) -> pd.DataFrame:
    result = _clean_string_columns(frame).rename(columns={"winner": "winner_raw"})
    result["season"] = _to_int(result["season"])
    result["game_number"] = _to_int(result["game_number"])
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
    result["duration_minutes"] = _to_float(result["duration_minutes"])
    _add_team_id(result, "team_a", "team_a_id", lookup)
    _add_team_id(result, "team_b", "team_b_id", lookup)
    result["winner_side"] = pd.Series(
        [
            _winner_side_from_raw(raw, team_a, team_b)
            for raw, team_a, team_b in zip(
                result["winner_raw"], result["team_a"], result["team_b"], strict=True
            )
        ],
        dtype="string",
    )
    result["winner_team_id"] = pd.Series(
        [
            _winner_team_id(side, team_a_id, team_b_id)
            for side, team_a_id, team_b_id in zip(
                result["winner_side"], result["team_a_id"], result["team_b_id"], strict=True
            )
        ],
        dtype="string",
    )
    return result


def _normalize_players(frame: pd.DataFrame, lookup: dict[tuple[int, str], str]) -> pd.DataFrame:
    result = _clean_string_columns(frame)
    result["season"] = _to_int(result["season"])
    _add_team_id(result, "team", "team_id", lookup)
    normalized_roles = result["role"].map(snake_case)
    result["role"] = normalized_roles.map(lambda value: ROLE_ALIASES.get(value, value)).astype(
        "string"
    )
    result["status"] = result["status"].map(snake_case).astype("string")
    return result


def _rank_bounds(value: Any) -> tuple[int | None, int | None]:
    text = clean_value(value)
    if text is None:
        return None, None
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def _to_bool(series: pd.Series) -> pd.Series:
    true_values = {"1", "true", "yes", "y"}
    false_values = {"0", "false", "no", "n"}

    def convert(value: Any) -> bool | None:
        text = clean_value(value)
        if text is None:
            return None
        if text.lower() in true_values:
            return True
        if text.lower() in false_values:
            return False
        return None

    return series.map(convert).astype("boolean")


def _normalize_championships(
    frame: pd.DataFrame, lookup: dict[tuple[int, str], str]
) -> pd.DataFrame:
    result = _clean_string_columns(frame)
    result["season"] = _to_int(result["season"])
    _add_team_id(result, "team", "team_id", lookup)
    bounds = result["final_rank"].map(_rank_bounds)
    result["final_rank_min"] = pd.Series((item[0] for item in bounds), dtype="Int64")
    result["final_rank_max"] = pd.Series((item[1] for item in bounds), dtype="Int64")
    result["champion"] = _to_bool(result["champion"])
    result["runner_up"] = _to_bool(result["runner_up"])
    return result


def _normalize_drafts(frame: pd.DataFrame, lookup: dict[tuple[int, str], str]) -> pd.DataFrame:
    result = _clean_string_columns(frame)
    result["season"] = _to_int(result["season"])
    result["game_number"] = _to_int(result["game_number"])
    result["order"] = _to_int(result["order"])
    result["action"] = (
        result["action"]
        .map(snake_case)
        .map(lambda value: ACTION_ALIASES.get(value, value))
        .astype("string")
    )
    if "scope" in result:
        result["scope"] = result["scope"].map(snake_case).astype("string")
    for column in ("total_count", "wins", "losses"):
        if column in result:
            result[column] = _to_int(result[column])
    if "win_rate_pct" in result:
        result["win_rate_pct"] = _to_float(result["win_rate_pct"])
    _add_team_id(result, "team", "team_id", lookup)
    result["is_aggregate"] = (
        result["team"].eq("ALL_TEAMS")
        | result["match_id"].str.contains("AGGREGATE", na=False)
        | result["action"].str.startswith("aggregate_", na=False)
    ).astype("boolean")
    return result


def _normalize_player_stats(
    frame: pd.DataFrame, lookup: dict[tuple[int, str], str]
) -> pd.DataFrame:
    result = _clean_string_columns(frame)
    result["season"] = _to_int(result["season"])
    _add_team_id(result, "team", "team_id", lookup)
    normalized_roles = result["role"].map(snake_case)
    result["role"] = normalized_roles.map(lambda value: ROLE_ALIASES.get(value, value)).astype(
        "string"
    )
    result["stat_scope"] = result["stat_scope"].map(snake_case).astype("string")
    result["coverage_status"] = result["coverage_status"].map(snake_case).astype("string")
    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"], format="%Y-%m-%d", errors="coerce"
    )
    integer_columns = (
        "ranking",
        "games_played",
        "total_kills",
        "total_deaths",
        "total_assists",
        "mvp_points",
        "signature_hero_games",
    )
    float_columns = (
        "win_rate_pct",
        "avg_kills",
        "avg_deaths",
        "avg_assists",
        "avg_kda",
        "kill_participation_pct",
        "signature_hero_win_rate_pct",
        "damage_taken_per_game",
        "damage_taken_per_minute",
    )
    for column in integer_columns:
        result[column] = _to_int(result[column])
    for column in float_columns:
        result[column] = _to_float(result[column])
    return result


def normalize_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Create normalized in-memory tables while preserving all raw CSV files."""
    raw = load_raw_tables(data_dir)
    teams = _normalize_teams(raw["teams"])
    lookup = _team_lookup(teams)
    return {
        "teams": teams,
        "matches": _normalize_matches(raw["matches"], lookup),
        "games": _normalize_games(raw["games"], lookup),
        "players": _normalize_players(raw["players"], lookup),
        "championships": _normalize_championships(raw["championships"], lookup),
        "drafts": _normalize_drafts(raw["drafts"], lookup),
        "player_season_stats": _normalize_player_stats(raw["player_season_stats"], lookup),
    }


def write_normalized_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    """Write one reproducible Parquet file for each normalized logical table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for table in TABLES:
        path = output_dir / f"{table}.parquet"
        tables[table].to_parquet(path, index=False)
        outputs[table] = path
    return outputs
