from pathlib import Path
from typing import Any

import pandas as pd

from mpl_predictor.analysis.common import dataframe_records, write_json

TABLE_KEYS = {
    "teams": ("season", "team_id"),
    "matches": ("match_id",),
    "games": ("match_id", "game_number"),
    "players": ("season", "team_id", "player_id"),
    "player_identity": ("player_id",),
    "championships": ("season", "team_id"),
    "drafts": ("season", "match_id", "game_number", "team_id", "action", "order"),
    "player_season_stats": ("season", "team_id", "player_id", "snapshot_label"),
}


def _pct(numerator: int | float, denominator: int | float) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _table_profile(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    key_columns = list(TABLE_KEYS[name])
    complete_keys = frame[key_columns].notna().all(axis=1)
    duplicate_key_rows = int(frame.loc[complete_keys].duplicated(key_columns, keep=False).sum())
    missing_cells = int(frame.isna().sum().sum())
    total_cells = int(frame.shape[0] * frame.shape[1])

    columns = {}
    for column in frame.columns:
        missing_count = int(frame[column].isna().sum())
        columns[column] = {
            "dtype": str(frame[column].dtype),
            "missing_count": missing_count,
            "missing_pct": _pct(missing_count, len(frame)),
            "unique_non_null": int(frame[column].nunique(dropna=True)),
        }

    return {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "key_columns": key_columns,
        "key_rows_with_null": int((~complete_keys).sum()),
        "duplicate_key_rows": duplicate_key_rows,
        "fully_duplicated_rows": int(frame.duplicated(keep=False).sum()),
        "missing_cell_count": missing_cells,
        "missing_cell_pct": _pct(missing_cells, total_cells),
        "columns": columns,
    }


def _season_coverage(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matches = tables["matches"]
    games = tables["games"]
    players = tables["players"]
    drafts = tables["drafts"]
    stats = tables["player_season_stats"]
    teams = tables["teams"]
    championships = tables["championships"]
    records: list[dict[str, Any]] = []

    for season in sorted(int(value) for value in teams["season"].unique()):
        season_matches = matches.loc[matches["season"].eq(season)]
        season_games = games.loc[games["season"].eq(season)]
        season_players = players.loc[players["season"].eq(season)]
        season_drafts = drafts.loc[drafts["season"].eq(season)]
        season_stats = stats.loc[stats["season"].eq(season)]

        game_keys = season_games[["match_id", "game_number"]].drop_duplicates()
        drafted_game_keys = season_drafts.loc[
            season_drafts[["match_id", "game_number"]].notna().all(axis=1),
            ["match_id", "game_number"],
        ].drop_duplicates()
        covered_games = game_keys.merge(
            drafted_game_keys, on=["match_id", "game_number"], how="inner"
        )

        records.append(
            {
                "season": season,
                "era": "franchise" if season >= 4 else "pre_franchise",
                "team_count": int(teams["season"].eq(season).sum()),
                "match_count": len(season_matches),
                "match_outcome_coverage_pct": _pct(
                    int(season_matches["winner_side"].notna().sum()), len(season_matches)
                ),
                "game_count": len(season_games),
                "game_outcome_coverage_pct": _pct(
                    int(season_games["winner_side"].notna().sum()), len(season_games)
                ),
                "game_duration_coverage_pct": _pct(
                    int(season_games["duration_minutes"].notna().sum()), len(season_games)
                ),
                "roster_player_count": len(season_players),
                "roster_role_coverage_pct": _pct(
                    int(season_players["role"].notna().sum()), len(season_players)
                ),
                "draft_row_count": len(season_drafts),
                "draft_game_coverage_pct": _pct(len(covered_games), len(game_keys)),
                "player_stat_row_count": len(season_stats),
                "champion_count": int(
                    championships.loc[championships["season"].eq(season), "champion"].sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _check(check_id: str, passed: bool, detail: str, count: int = 0) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "pass" if passed else "fail",
        "count": count,
        "detail": detail,
    }


def build_quality_report(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Profile canonical data and state which feature families are safe for modeling."""
    teams = tables["teams"]
    matches = tables["matches"]
    games = tables["games"]
    players = tables["players"]
    championships = tables["championships"]
    stats = tables["player_season_stats"]
    identities = tables["player_identity"]
    franchise_matches = matches.loc[matches["season"].ge(4)]
    franchise_teams = teams.loc[teams["season"].ge(4)]
    franchise_players = players.loc[players["season"].ge(4)]

    champion_counts = championships.groupby("season")["champion"].sum()
    core_checks = [
        _check(
            "unique_match_id",
            not matches["match_id"].duplicated().any(),
            "Setiap match_id harus unik.",
            int(matches["match_id"].duplicated(keep=False).sum()),
        ),
        _check(
            "complete_match_inputs",
            matches[
                ["date", "team_a_id", "team_b_id", "team_a_score", "team_b_score", "winner_side"]
            ]
            .notna()
            .all(axis=None),
            "Tanggal, peserta, skor, dan hasil pertandingan wajib lengkap.",
            int(
                matches[
                    [
                        "date",
                        "team_a_id",
                        "team_b_id",
                        "team_a_score",
                        "team_b_score",
                        "winner_side",
                    ]
                ]
                .isna()
                .sum()
                .sum()
            ),
        ),
        _check(
            "complete_franchise_slots",
            franchise_teams["franchise_slot_id"].notna().all(),
            "Semua tim Season 4+ harus mempunyai franchise_slot_id.",
            int(franchise_teams["franchise_slot_id"].isna().sum()),
        ),
        _check(
            "complete_team_identity",
            teams["organization_id"].notna().all(),
            "Semua team-season harus terhubung ke organization_id.",
            int(teams["organization_id"].isna().sum()),
        ),
        _check(
            "one_champion_per_season",
            champion_counts.eq(1).all() and len(champion_counts) == teams["season"].nunique(),
            "Setiap musim harus memiliki tepat satu label juara.",
            int(champion_counts.ne(1).sum()),
        ),
        _check(
            "complete_player_identity",
            players["player_id"].notna().all(),
            "Semua baris roster harus terhubung ke player_id.",
            int(players["player_id"].isna().sum()),
        ),
    ]

    game_outcome_coverage = _pct(int(games["winner_side"].notna().sum()), len(games))
    duration_coverage = _pct(int(games["duration_minutes"].notna().sum()), len(games))
    role_coverage = _pct(int(franchise_players["role"].notna().sum()), len(franchise_players))
    reviewed_player_count = int(identities["identity_review_required"].sum())
    franchise_team_seasons = len(franchise_teams)
    stat_team_seasons = len(
        stats.loc[stats["season"].ge(4), ["season", "team_id"]].drop_duplicates()
    )

    feature_groups = [
        {
            "feature_group": "team_match_history",
            "source_tables": ["matches"],
            "status": "ready",
            "coverage": "100% skor dan hasil match tersedia",
            "decision": "Fitur inti untuk baseline, Elo, dan performa rolling.",
            "leakage_control": "Hanya pertandingan dengan completion date <= cutoff snapshot.",
        },
        {
            "feature_group": "team_and_franchise_identity",
            "source_tables": ["teams"],
            "status": "ready",
            "coverage": "100% organization; 100% franchise slot pada Season 4+",
            "decision": "Gunakan franchise_slot_id untuk kesinambungan lintas rebrand.",
            "leakage_control": "Nama dan ID menjadi identifier, bukan sinyal target langsung.",
        },
        {
            "feature_group": "roster_identity_and_roles",
            "source_tables": ["players", "player_identity"],
            "status": "conditional",
            "coverage": (
                f"Role Season 4+ terisi {role_coverage}%; "
                f"{reviewed_player_count} identitas perlu review"
            ),
            "decision": "Boleh untuk continuity/experience setelah ada tanggal efektif roster.",
            "leakage_control": (
                "Roster historis saat ini adalah daftar semusim dan belum memiliki "
                "valid_from/valid_to."
            ),
        },
        {
            "feature_group": "game_outcomes",
            "source_tables": ["games"],
            "status": "conditional",
            "coverage": f"Hasil game terisi {game_outcome_coverage}% secara keseluruhan",
            "decision": (
                "Gunakan score game dari matches sebagai sumber utama; detail games hanya "
                "saat tersedia."
            ),
            "leakage_control": (
                "Tambahkan indikator missing dan jangan menganggap missing sebagai kalah/menang."
            ),
        },
        {
            "feature_group": "game_duration",
            "source_tables": ["games"],
            "status": "experimental",
            "coverage": f"Durasi game terisi {duration_coverage}% secara keseluruhan",
            "decision": (
                "Tidak menjadi fitur baseline; hanya eksperimen pada musim dengan coverage memadai."
            ),
            "leakage_control": (
                "Evaluasi terpisah per era agar pola missing tidak menjadi proxy musim."
            ),
        },
        {
            "feature_group": "draft_and_hero",
            "source_tables": ["drafts"],
            "status": "experimental",
            "coverage": (
                "Coverage sangat berbeda antar musim dan sangat tipis pada beberapa musim terbaru."
            ),
            "decision": (
                "Tidak menjadi fitur baseline; gunakan hanya pada eksperimen coverage-terkontrol."
            ),
            "leakage_control": "Draft hanya boleh masuk setelah game terkait selesai pada cutoff.",
        },
        {
            "feature_group": "published_player_statistics",
            "source_tables": ["player_season_stats"],
            "status": "excluded_as_of",
            "coverage": (
                f"{len(stats)} baris pada "
                f"{stat_team_seasons}/{franchise_team_seasons} team-season; "
                "merupakan pemain terpilih/snapshot yang tidak seragam"
            ),
            "decision": "Jangan dipakai sebagai agregat roster atau fitur inti model as-of.",
            "leakage_control": (
                "Banyak snapshot dipublikasikan pada/selepas akhir musim dan dapat "
                "membocorkan target."
            ),
        },
        {
            "feature_group": "champion_and_final_rank",
            "source_tables": ["championships"],
            "status": "target_only",
            "coverage": "Tepat satu juara pada setiap Season 1-17",
            "decision": "champion adalah label; final_rank hanya untuk evaluasi/deskripsi.",
            "leakage_control": (
                "Seluruh kolom tabel ini dilarang menjadi fitur prediksi musim yang sama."
            ),
        },
    ]

    advisory_findings = [
        {
            "finding_id": "pre_franchise_scope",
            "severity": "info",
            "detail": "Season 1-3 tetap disimpan untuk konteks, bukan data utama model franchise.",
        },
        {
            "finding_id": "roster_temporal_fields_missing",
            "severity": "warning",
            "detail": "players belum mempunyai valid_from/valid_to atau tanggal pengumuman roster.",
        },
        {
            "finding_id": "player_stats_selection_bias",
            "severity": "warning",
            "detail": (
                "player_season_stats berisi statistik terpilih dan definisi snapshot yang berbeda."
            ),
        },
        {
            "finding_id": "partial_game_details",
            "severity": "warning",
            "detail": (
                "Hasil dan durasi pada games parsial; skor series pada matches lebih konsisten."
            ),
        },
        {
            "finding_id": "partial_draft_coverage",
            "severity": "warning",
            "detail": "Draft tidak mempunyai coverage yang konsisten antar musim.",
        },
        {
            "finding_id": "non_feature_columns",
            "severity": "info",
            "detail": (
                "Index, source, source_file, URL, nama mentah, dan ID target tidak digunakan "
                "sebagai fitur numerik."
            ),
        },
    ]

    return {
        "report_version": "1.0",
        "dataset_scope": {
            "season_min": int(teams["season"].min()),
            "season_max": int(teams["season"].max()),
            "primary_modeling_seasons": [4, int(teams["season"].max())],
            "table_count": len(tables),
            "total_rows": sum(len(frame) for frame in tables.values()),
            "team_season_count": len(teams),
            "franchise_team_season_count": franchise_team_seasons,
            "match_count": len(matches),
            "franchise_match_count": len(franchise_matches),
            "champion_label_count": int(championships["champion"].sum()),
        },
        "core_modeling_ready": all(check["status"] == "pass" for check in core_checks),
        "blocking_issue_count": sum(check["status"] == "fail" for check in core_checks),
        "core_checks": core_checks,
        "feature_groups": feature_groups,
        "advisory_findings": advisory_findings,
        "season_coverage": dataframe_records(_season_coverage(tables)),
        "tables": {name: _table_profile(name, frame) for name, frame in sorted(tables.items())},
    }


def write_quality_report(report: dict[str, Any], path: Path) -> None:
    write_json(report, path)
