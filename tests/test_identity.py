from pathlib import Path

import pandas as pd

from mpl_predictor.data.identity import (
    build_canonical_tables,
    identity_summary,
    load_player_alias_overrides,
    load_team_identity_rules,
    write_canonical_tables,
)
from mpl_predictor.data.normalization import normalize_tables

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RULES_PATH = PROJECT_ROOT / "config" / "team_identity_rules.csv"
PLAYER_ALIASES_PATH = PROJECT_ROOT / "config" / "player_alias_overrides.csv"


def _canonical_tables() -> dict[str, pd.DataFrame]:
    normalized = normalize_tables(DATA_DIR)
    rules = load_team_identity_rules(RULES_PATH)
    aliases = load_player_alias_overrides(PLAYER_ALIASES_PATH)
    return build_canonical_tables(normalized, rules, aliases)


def test_team_identity_rules_cover_every_team_season() -> None:
    canonical = _canonical_tables()
    teams = canonical["teams"]

    assert len(teams) == 150
    assert teams["organization_id"].notna().all()
    assert teams.loc[teams["season"].lt(4), "franchise_slot_id"].isna().all()
    assert teams.loc[teams["season"].ge(4), "franchise_slot_id"].notna().all()
    assert (
        teams.loc[teams["season"].between(4, 11)]
        .groupby("season")["franchise_slot_id"]
        .nunique()
        .eq(8)
        .all()
    )
    assert (
        teams.loc[teams["season"].ge(12)]
        .groupby("season")["franchise_slot_id"]
        .nunique()
        .eq(9)
        .all()
    )


def test_verified_brand_changes_keep_the_correct_slot() -> None:
    teams = _canonical_tables()["teams"].set_index(["season", "team_id"])

    assert (
        teams.loc[(7, "GFLX"), "franchise_slot_id"]
        == teams.loc[(8, "RBG"), "franchise_slot_id"]
        == teams.loc[(15, "NAVI"), "franchise_slot_id"]
    )
    assert (
        teams.loc[(12, "AURA"), "franchise_slot_id"] == teams.loc[(13, "TLID"), "franchise_slot_id"]
    )
    assert (
        teams.loc[(12, "ONIC"), "franchise_slot_id"]
        == teams.loc[(13, "FNOC"), "franchise_slot_id"]
        == teams.loc[(15, "ONIC"), "franchise_slot_id"]
    )
    assert teams.loc[(13, "FNOC"), "organization_id"] == "ONIC"
    assert teams.loc[(15, "NAVI"), "organization_id"] == "NAVI"


def test_player_identity_is_complete_and_conservative() -> None:
    canonical = _canonical_tables()
    players = canonical["player_identity"]
    summary = identity_summary(canonical)

    assert len(players) == 375
    assert summary["player_alias_group_count"] >= 74
    assert summary["ambiguous_same_season_player_count"] == 0
    assert summary["unmapped_roster_player_count"] == 0
    assert summary["unmapped_player_stat_count"] == 0
    assert summary["player_review_required_count"] == 6
    assert "player_identity_review_required" in canonical["players"].columns

    aboy_rows = canonical["players"].loc[
        canonical["players"]["player"].isin(["A B O Y", "ABOY", "Aboy"])
    ]
    assert aboy_rows["player_id"].nunique() == 1

    moreno_rows = canonical["players"].loc[
        canonical["players"]["player"].isin(["Moreno", "Morenoo", "MORENO", "MORENOOO"])
    ]
    assert moreno_rows["player_id"].nunique() == 1

    season_13_vyn = canonical["players"].loc[
        canonical["players"]["season"].eq(13) & canonical["players"]["player"].isin(["VYN", "Vynn"])
    ]
    assert season_13_vyn["player_id"].nunique() == 2


def test_canonical_tables_preserve_fact_row_counts(tmp_path: Path) -> None:
    canonical = _canonical_tables()
    outputs = write_canonical_tables(canonical, tmp_path)

    assert len(outputs) == 8
    assert len(canonical["matches"]) == 1182
    assert len(canonical["games"]) == 3013
    assert len(canonical["players"]) == 1079
    assert len(canonical["drafts"]) == 35212
    assert all(path.exists() for path in outputs.values())
