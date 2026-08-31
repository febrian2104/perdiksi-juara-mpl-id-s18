import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from mpl_predictor.analysis.common import load_canonical_tables
from mpl_predictor.analysis.eda import build_eda_report, write_eda_figures, write_eda_report
from mpl_predictor.analysis.prediction_policy import (
    build_prediction_policy_report,
    build_prediction_windows,
    load_prediction_policy,
    write_prediction_outputs,
)
from mpl_predictor.analysis.quality import build_quality_report, write_quality_report
from mpl_predictor.config import get_project_paths
from mpl_predictor.data.audit import AuditReport, audit_data
from mpl_predictor.data.identity import (
    build_canonical_tables,
    identity_summary,
    load_player_alias_overrides,
    load_team_identity_rules,
    write_canonical_tables,
    write_identity_summary,
)
from mpl_predictor.data.normalization import normalize_tables, write_normalized_tables
from mpl_predictor.data.season18 import (
    TEAM_METADATA,
    build_season18_report,
    build_season18_teams,
    fetch_official_html,
    merge_roster_history,
    parse_roster_html,
    parse_schedule_html,
    write_season18_outputs,
)
from mpl_predictor.data.semantic_audit import (
    SemanticAuditReport,
    audit_semantics,
    write_semantic_report,
)
from mpl_predictor.features.matches import (
    build_match_feature_report,
    build_match_features,
    write_match_feature_outputs,
)
from mpl_predictor.features.snapshots import (
    build_feature_report,
    build_snapshot_features,
    load_feature_config,
    write_snapshot_outputs,
)
from mpl_predictor.models.baseline import (
    build_baseline_predictions,
    build_baseline_report,
    write_baseline_outputs,
)
from mpl_predictor.models.evaluation import (
    build_model_evaluation_report,
    write_evaluation_figures,
    write_model_outputs,
)
from mpl_predictor.models.explainability import (
    build_explainability_report,
    build_global_importance,
    build_match_explanations,
    build_team_explanations,
    write_explainability_outputs,
)
from mpl_predictor.models.final import (
    build_final_model_report,
    load_final_match_model,
    train_final_match_model,
    write_final_model_outputs,
)
from mpl_predictor.models.season18_predictions import (
    build_season18_prediction_history,
    load_season18_tables,
    write_season18_prediction_history,
)
from mpl_predictor.models.simulation import (
    build_season18_match_probabilities,
    load_simulation_config,
    simulate_season18,
    write_simulation_outputs,
)
from mpl_predictor.models.walk_forward import (
    load_model_config,
    walk_forward_champion_predictions,
    walk_forward_match_predictions,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpl-predictor",
        description="Utilities for the MPL Indonesia champion prediction project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit historical CSV datasets.")
    audit_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory; defaults to <project>/data.",
    )
    audit_parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    audit_parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return a failing exit code when warnings are present.",
    )

    semantic_parser = subparsers.add_parser(
        "semantic-audit", help="Validate cross-table meaning and report data coverage."
    )
    semantic_parser.add_argument("--data-dir", type=Path, default=None)
    semantic_parser.add_argument("--output", type=Path, default=None)
    semantic_parser.add_argument("--json", action="store_true")
    semantic_parser.add_argument("--fail-on-warning", action="store_true")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Write normalized Parquet tables and a semantic audit report."
    )
    normalize_parser.add_argument("--data-dir", type=Path, default=None)
    normalize_parser.add_argument("--output-dir", type=Path, default=None)
    normalize_parser.add_argument("--report", type=Path, default=None)

    canonical_parser = subparsers.add_parser(
        "canonicalize", help="Build canonical team, player, and historical tables."
    )
    canonical_parser.add_argument("--data-dir", type=Path, default=None)
    canonical_parser.add_argument("--rules", type=Path, default=None)
    canonical_parser.add_argument("--player-aliases", type=Path, default=None)
    canonical_parser.add_argument("--output-dir", type=Path, default=None)
    canonical_parser.add_argument("--report", type=Path, default=None)

    quality_parser = subparsers.add_parser(
        "quality-report", help="Profile canonical datasets and assess feature readiness."
    )
    quality_parser.add_argument("--canonical-dir", type=Path, default=None)
    quality_parser.add_argument("--output", type=Path, default=None)

    eda_parser = subparsers.add_parser(
        "eda", help="Generate modeling-oriented EDA summaries and figures."
    )
    eda_parser.add_argument("--canonical-dir", type=Path, default=None)
    eda_parser.add_argument("--output", type=Path, default=None)
    eda_parser.add_argument("--figures-dir", type=Path, default=None)
    eda_parser.add_argument(
        "--no-figures", action="store_true", help="Only write the JSON EDA report."
    )

    prediction_parser = subparsers.add_parser(
        "prediction-policy", help="Generate preseason and weekly historical cutoff windows."
    )
    prediction_parser.add_argument("--canonical-dir", type=Path, default=None)
    prediction_parser.add_argument("--policy", type=Path, default=None)
    prediction_parser.add_argument("--output", type=Path, default=None)
    prediction_parser.add_argument("--windows-output", type=Path, default=None)

    feature_parser = subparsers.add_parser(
        "build-features", help="Build leakage-safe team features for every historical snapshot."
    )
    feature_parser.add_argument("--canonical-dir", type=Path, default=None)
    feature_parser.add_argument("--policy", type=Path, default=None)
    feature_parser.add_argument("--config", type=Path, default=None)
    feature_parser.add_argument("--output", type=Path, default=None)
    feature_parser.add_argument("--report", type=Path, default=None)

    baseline_parser = subparsers.add_parser(
        "baseline", help="Generate and evaluate uniform and Elo champion baselines."
    )
    baseline_parser.add_argument("--features", type=Path, default=None)
    baseline_parser.add_argument("--config", type=Path, default=None)
    baseline_parser.add_argument("--output", type=Path, default=None)
    baseline_parser.add_argument("--report", type=Path, default=None)

    match_feature_parser = subparsers.add_parser(
        "build-match-features", help="Build pre-match features for the win-probability model."
    )
    match_feature_parser.add_argument("--canonical-dir", type=Path, default=None)
    match_feature_parser.add_argument("--feature-config", type=Path, default=None)
    match_feature_parser.add_argument("--output", type=Path, default=None)
    match_feature_parser.add_argument("--report", type=Path, default=None)

    backtest_parser = subparsers.add_parser(
        "backtest", help="Run match and champion walk-forward models with past-only calibration."
    )
    backtest_parser.add_argument("--match-features", type=Path, default=None)
    backtest_parser.add_argument("--snapshot-features", type=Path, default=None)
    backtest_parser.add_argument("--baselines", type=Path, default=None)
    backtest_parser.add_argument("--config", type=Path, default=None)
    backtest_parser.add_argument("--match-output", type=Path, default=None)
    backtest_parser.add_argument("--champion-output", type=Path, default=None)
    backtest_parser.add_argument("--report", type=Path, default=None)
    backtest_parser.add_argument("--figures-dir", type=Path, default=None)
    backtest_parser.add_argument("--no-figures", action="store_true")

    season18_parser = subparsers.add_parser(
        "sync-season18", help="Fetch and validate official Season 18 teams, rosters, and schedule."
    )
    season18_parser.add_argument(
        "--observed-at",
        type=date.fromisoformat,
        default=None,
        help="Observation date in YYYY-MM-DD; defaults to today.",
    )
    season18_parser.add_argument("--player-aliases", type=Path, default=None)
    season18_parser.add_argument("--output-dir", type=Path, default=None)
    season18_parser.add_argument("--report", type=Path, default=None)

    final_parser = subparsers.add_parser(
        "train-final", help="Select and train the final calibrated match model."
    )
    final_parser.add_argument("--match-features", type=Path, default=None)
    final_parser.add_argument("--config", type=Path, default=None)
    final_parser.add_argument("--evaluation-report", type=Path, default=None)
    final_parser.add_argument("--artifact", type=Path, default=None)
    final_parser.add_argument("--report", type=Path, default=None)

    simulation_parser = subparsers.add_parser(
        "simulate-season18", help="Simulate the remaining S18 regular season and playoffs."
    )
    simulation_parser.add_argument("--canonical-dir", type=Path, default=None)
    simulation_parser.add_argument("--season18-dir", type=Path, default=None)
    simulation_parser.add_argument("--feature-config", type=Path, default=None)
    simulation_parser.add_argument("--model-artifact", type=Path, default=None)
    simulation_parser.add_argument("--config", type=Path, default=None)
    simulation_parser.add_argument("--iterations", type=int, default=None)
    simulation_parser.add_argument("--simulation-output", type=Path, default=None)
    simulation_parser.add_argument("--match-output", type=Path, default=None)
    simulation_parser.add_argument("--report", type=Path, default=None)

    update_parser = subparsers.add_parser(
        "update-season18-predictions",
        help="Reconstruct preseason and update every completed S18 weekly snapshot.",
    )
    update_parser.add_argument("--canonical-dir", type=Path, default=None)
    update_parser.add_argument("--season18-dir", type=Path, default=None)
    update_parser.add_argument("--feature-config", type=Path, default=None)
    update_parser.add_argument("--model-artifact", type=Path, default=None)
    update_parser.add_argument("--config", type=Path, default=None)
    update_parser.add_argument("--iterations", type=int, default=None)
    update_parser.add_argument("--prediction-dir", type=Path, default=None)
    update_parser.add_argument("--report", type=Path, default=None)
    update_parser.add_argument("--latest-report", type=Path, default=None)

    explain_parser = subparsers.add_parser(
        "explain-season18", help="Generate global, match, and team prediction explanations."
    )
    explain_parser.add_argument("--model-artifact", type=Path, default=None)
    explain_parser.add_argument("--predictions", type=Path, default=None)
    explain_parser.add_argument("--match-probabilities", type=Path, default=None)
    explain_parser.add_argument("--prediction-dir", type=Path, default=None)
    explain_parser.add_argument("--report", type=Path, default=None)
    return parser


def _print_human_report(report: AuditReport) -> None:
    print("MPL historical data audit")
    print(f"Files: {len(report.files)}")
    print(f"Rows: {report.total_rows}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")

    if report.issues:
        print("\nIssues:")
        for issue in report.issues:
            location = f" [{issue.path}]" if issue.path else ""
            print(f"- {issue.severity.upper()} {issue.code}: {issue.message}{location}")


def _print_semantic_report(report: SemanticAuditReport) -> None:
    print("MPL semantic data audit")
    print(f"Rows: {sum(report.row_counts.values())}")
    print(f"Passed checks: {sum(check.status == 'pass' for check in report.checks)}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")
    print(f"Information: {len(report.information)}")
    for check in report.checks:
        if check.status != "pass":
            print(f"- {check.status.upper()} {check.code} ({check.count}): {check.message}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        paths = get_project_paths()
        data_dir = (args.data_dir or paths.data).resolve()
        report = audit_data(data_dir)
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        else:
            _print_human_report(report)
        return int(bool(report.errors or (args.fail_on_warning and report.warnings)))

    if args.command in {"semantic-audit", "normalize"}:
        paths = get_project_paths()
        data_dir = (args.data_dir or paths.data).resolve()
        tables = normalize_tables(data_dir)
        report = audit_semantics(tables)

        if args.command == "semantic-audit":
            report_path = (args.output or paths.reports / "semantic_audit.json").resolve()
            write_semantic_report(report, report_path)
            if args.json:
                print(json.dumps(report.as_dict(), indent=2))
            else:
                _print_semantic_report(report)
                print(f"Report: {report_path}")
            return int(bool(report.errors or (args.fail_on_warning and report.warnings)))

        output_dir = (args.output_dir or paths.interim / "normalized").resolve()
        report_path = (args.report or paths.reports / "semantic_audit.json").resolve()
        outputs = write_normalized_tables(tables, output_dir)
        write_semantic_report(report, report_path)
        print(f"Normalized tables: {len(outputs)}")
        print(f"Normalized rows: {sum(len(frame) for frame in tables.values())}")
        print(f"Output directory: {output_dir}")
        print(f"Semantic report: {report_path}")
        print(f"Semantic errors: {len(report.errors)}")
        print(f"Semantic warnings: {len(report.warnings)}")
        return int(bool(report.errors))

    if args.command == "canonicalize":
        paths = get_project_paths()
        data_dir = (args.data_dir or paths.data).resolve()
        rules_path = (args.rules or paths.root / "config" / "team_identity_rules.csv").resolve()
        player_alias_path = (
            args.player_aliases or paths.root / "config" / "player_alias_overrides.csv"
        ).resolve()
        output_dir = (args.output_dir or paths.processed / "canonical").resolve()
        report_path = (args.report or paths.reports / "identity_mapping_summary.json").resolve()

        normalized_tables = normalize_tables(data_dir)
        semantic_report = audit_semantics(normalized_tables)
        if semantic_report.errors:
            print("Canonicalization stopped because semantic audit errors were found.")
            return 1

        rules = load_team_identity_rules(rules_path)
        player_aliases = load_player_alias_overrides(player_alias_path)
        canonical_tables = build_canonical_tables(normalized_tables, rules, player_aliases)
        summary = identity_summary(canonical_tables)
        outputs = write_canonical_tables(canonical_tables, output_dir)
        write_identity_summary(summary, report_path)

        blocking_fields = (
            "franchise_rows_without_slot",
            "pre_franchise_rows_with_slot",
            "ambiguous_same_season_player_count",
            "unmapped_roster_player_count",
            "unmapped_player_stat_count",
        )
        blocking_count = sum(int(summary[field]) for field in blocking_fields)
        print(f"Canonical tables: {len(outputs)}")
        print(f"Team-season identities: {summary['team_season_rows']}")
        print(f"Organizations: {summary['organization_count']}")
        print(f"Franchise slots: {summary['franchise_slot_count']}")
        print(f"Player identities: {summary['player_identity_count']}")
        print(f"Player aliases: {summary['player_alias_group_count']}")
        print(f"Player identities requiring review: {summary['player_review_required_count']}")
        print(f"Output directory: {output_dir}")
        print(f"Identity report: {report_path}")
        return int(blocking_count > 0)

    if args.command == "quality-report":
        paths = get_project_paths()
        canonical_dir = (args.canonical_dir or paths.processed / "canonical").resolve()
        report_path = (args.output or paths.reports / "dataset_quality_report.json").resolve()
        tables = load_canonical_tables(canonical_dir)
        report = build_quality_report(tables)
        write_quality_report(report, report_path)
        print("MPL canonical dataset and feature quality")
        print(f"Tables: {report['dataset_scope']['table_count']}")
        print(f"Rows: {report['dataset_scope']['total_rows']}")
        print(f"Core modeling ready: {report['core_modeling_ready']}")
        print(f"Blocking issues: {report['blocking_issue_count']}")
        print(f"Report: {report_path}")
        return int(report["blocking_issue_count"] > 0)

    if args.command == "eda":
        paths = get_project_paths()
        canonical_dir = (args.canonical_dir or paths.processed / "canonical").resolve()
        report_path = (args.output or paths.reports / "eda_summary.json").resolve()
        figures_dir = (args.figures_dir or paths.figures).resolve()
        tables = load_canonical_tables(canonical_dir)
        report, frames = build_eda_report(tables)
        write_eda_report(report, report_path)
        figure_outputs = [] if args.no_figures else write_eda_figures(frames, figures_dir)
        print("MPL exploratory data analysis")
        print(f"Team-season observations: {report['scope']['team_season_observations']}")
        print(
            f"Franchise champion observations: {report['scope']['franchise_champion_observations']}"
        )
        print(f"Figures: {len(figure_outputs)}")
        print(f"Report: {report_path}")
        if figure_outputs:
            print(f"Figures directory: {figures_dir}")
        return 0

    if args.command == "prediction-policy":
        paths = get_project_paths()
        canonical_dir = (args.canonical_dir or paths.processed / "canonical").resolve()
        policy_path = (args.policy or paths.root / "config" / "prediction_policy.json").resolve()
        report_path = (args.output or paths.reports / "prediction_policy_summary.json").resolve()
        windows_path = (args.windows_output or paths.reports / "prediction_windows.csv").resolve()
        tables = load_canonical_tables(canonical_dir)
        policy = load_prediction_policy(policy_path)
        windows = build_prediction_windows(tables, policy)
        report = build_prediction_policy_report(policy, windows)
        write_prediction_outputs(report, windows, report_path, windows_path)
        print("MPL prediction timing policy")
        print(f"Historical snapshots: {report['historical_windows']['snapshot_count']}")
        print(
            "Preseason / weekly: "
            f"{report['historical_windows']['preseason_snapshot_count']} / "
            f"{report['historical_windows']['weekly_snapshot_count']}"
        )
        print(f"Policy report: {report_path}")
        print(f"Historical windows: {windows_path}")
        return 0

    if args.command == "build-features":
        paths = get_project_paths()
        canonical_dir = (args.canonical_dir or paths.processed / "canonical").resolve()
        policy_path = (args.policy or paths.root / "config" / "prediction_policy.json").resolve()
        config_path = (args.config or paths.root / "config" / "feature_config.json").resolve()
        output_path = (
            args.output or paths.processed / "features" / "team_snapshot_features.parquet"
        ).resolve()
        report_path = (args.report or paths.reports / "feature_engineering_report.json").resolve()
        tables = load_canonical_tables(canonical_dir)
        policy = load_prediction_policy(policy_path)
        feature_config = load_feature_config(config_path)
        windows = build_prediction_windows(tables, policy)
        features, roster_metadata = build_snapshot_features(tables, windows, feature_config)
        report = build_feature_report(features, windows, roster_metadata)
        write_snapshot_outputs(features, report, output_path, report_path)
        print("MPL snapshot feature engineering")
        print(f"Snapshots: {report['snapshot_count']}")
        print(f"Team-snapshot rows: {report['feature_row_count']}")
        print(
            "Enabled / defined feature columns: "
            f"{report['enabled_feature_column_count']} / {report['feature_column_count']}"
        )
        print(
            f"Current roster temporal enabled: {roster_metadata['current_roster_features_enabled']}"
        )
        print(f"Blocking issues: {report['blocking_issue_count']}")
        print(f"Features: {output_path}")
        print(f"Report: {report_path}")
        return int(report["blocking_issue_count"] > 0)

    if args.command == "baseline":
        paths = get_project_paths()
        feature_path = (
            args.features or paths.processed / "features" / "team_snapshot_features.parquet"
        ).resolve()
        config_path = (args.config or paths.root / "config" / "feature_config.json").resolve()
        output_path = (
            args.output or paths.processed / "predictions" / "baseline_predictions.parquet"
        ).resolve()
        report_path = (args.report or paths.reports / "baseline_report.json").resolve()
        features = pd.read_parquet(feature_path)
        feature_config = load_feature_config(config_path)
        predictions = build_baseline_predictions(features, feature_config)
        report = build_baseline_report(predictions, feature_config)
        write_baseline_outputs(predictions, report, output_path, report_path)
        print("MPL probability and Elo baselines")
        print(f"Prediction rows: {report['prediction_row_count']}")
        print(f"Evaluated snapshots: {report['evaluated_snapshot_count']}")
        print(
            "Invalid probability sums: "
            f"{report['probability_validation']['invalid_sum_group_count']}"
        )
        print(f"Predictions: {output_path}")
        print(f"Report: {report_path}")
        return int(report["probability_validation"]["invalid_sum_group_count"] > 0)

    if args.command == "build-match-features":
        paths = get_project_paths()
        canonical_dir = (args.canonical_dir or paths.processed / "canonical").resolve()
        feature_config_path = (
            args.feature_config or paths.root / "config" / "feature_config.json"
        ).resolve()
        output_path = (
            args.output or paths.processed / "features" / "match_features.parquet"
        ).resolve()
        report_path = (args.report or paths.reports / "match_feature_report.json").resolve()
        tables = load_canonical_tables(canonical_dir)
        feature_config = load_feature_config(feature_config_path)
        features = build_match_features(tables, feature_config)
        report = build_match_feature_report(features)
        write_match_feature_outputs(features, report, output_path, report_path)
        print("MPL pre-match feature engineering")
        print(f"Match rows: {report['match_row_count']}")
        print(f"Feature columns: {len(report['feature_columns'])}")
        print(f"Blocking issues: {report['blocking_issue_count']}")
        print(f"Features: {output_path}")
        print(f"Report: {report_path}")
        return int(report["blocking_issue_count"] > 0)

    if args.command == "backtest":
        paths = get_project_paths()
        match_feature_path = (
            args.match_features or paths.processed / "features" / "match_features.parquet"
        ).resolve()
        snapshot_feature_path = (
            args.snapshot_features
            or paths.processed / "features" / "team_snapshot_features.parquet"
        ).resolve()
        baseline_path = (
            args.baselines or paths.processed / "predictions" / "baseline_predictions.parquet"
        ).resolve()
        config_path = (args.config or paths.root / "config" / "model_config.json").resolve()
        match_output_path = (
            args.match_output
            or paths.processed / "predictions" / "match_walk_forward_predictions.parquet"
        ).resolve()
        champion_output_path = (
            args.champion_output
            or paths.processed / "predictions" / "champion_walk_forward_predictions.parquet"
        ).resolve()
        report_path = (args.report or paths.reports / "model_evaluation_report.json").resolve()
        figures_dir = (args.figures_dir or paths.figures).resolve()

        match_features = pd.read_parquet(match_feature_path)
        snapshot_features = pd.read_parquet(snapshot_feature_path)
        baseline_predictions = pd.read_parquet(baseline_path)
        model_config = load_model_config(config_path)
        match_predictions, match_folds = walk_forward_match_predictions(
            match_features, model_config
        )
        champion_predictions, champion_folds = walk_forward_champion_predictions(
            snapshot_features, baseline_predictions, model_config
        )
        report, frames = build_model_evaluation_report(
            match_predictions,
            champion_predictions,
            match_folds,
            champion_folds,
            model_config,
        )
        write_model_outputs(
            match_predictions,
            champion_predictions,
            report,
            match_output_path,
            champion_output_path,
            report_path,
        )
        figure_outputs = [] if args.no_figures else write_evaluation_figures(frames, figures_dir)
        print("MPL walk-forward model evaluation")
        print(f"Evaluated matches: {report['evaluation_scope']['match_count']}")
        print(f"Evaluated snapshots: {report['evaluation_scope']['snapshot_count']}")
        print(
            "Invalid champion probability sums: "
            f"{report['probability_validation']['invalid_probability_sum_count']}"
        )
        print(f"Figures: {len(figure_outputs)}")
        print(f"Match predictions: {match_output_path}")
        print(f"Champion predictions: {champion_output_path}")
        print(f"Report: {report_path}")
        return int(report["probability_validation"]["invalid_probability_sum_count"] > 0)

    if args.command == "sync-season18":
        paths = get_project_paths()
        observed_at = args.observed_at or date.today()
        alias_path = (
            args.player_aliases or paths.root / "config" / "player_alias_overrides.csv"
        ).resolve()
        output_dir = (args.output_dir or paths.data / "season18").resolve()
        report_path = (args.report or paths.reports / "season18_data_report.json").resolve()
        player_aliases = load_player_alias_overrides(alias_path)
        teams = build_season18_teams()
        schedule = parse_schedule_html(
            fetch_official_html("https://id-mpl.com/id/schedule"), observed_at
        )
        roster_frames = []
        for team_id, metadata in TEAM_METADATA.items():
            url = f"https://id-mpl.com/en/team/{metadata['slug']}"
            roster_frames.append(
                parse_roster_html(fetch_official_html(url), team_id, observed_at, player_aliases)
            )
        rosters = pd.concat(roster_frames, ignore_index=True)
        existing_roster_path = output_dir / "rosters.csv"
        if existing_roster_path.exists():
            existing_rosters = pd.read_csv(existing_roster_path)
            rosters = merge_roster_history(existing_rosters, rosters, observed_at)
        report = build_season18_report(teams, rosters, schedule, observed_at)
        write_season18_outputs(teams, rosters, schedule, report, output_dir, report_path)
        print("MPL Indonesia Season 18 official data integration")
        print(f"Observation date: {observed_at.isoformat()}")
        print(f"Teams / roster members: {len(teams)} / {len(rosters)}")
        print(
            "Completed / scheduled matches: "
            f"{report['scope']['completed_match_count']} / "
            f"{report['scope']['remaining_match_count']}"
        )
        print(f"Blocking issues: {report['blocking_issue_count']}")
        print(f"Data: {output_dir}")
        print(f"Report: {report_path}")
        return int(report["blocking_issue_count"] > 0)

    if args.command == "train-final":
        paths = get_project_paths()
        match_path = (
            args.match_features or paths.processed / "features" / "match_features.parquet"
        ).resolve()
        config_path = (args.config or paths.root / "config" / "model_config.json").resolve()
        evaluation_path = (
            args.evaluation_report or paths.reports / "model_evaluation_report.json"
        ).resolve()
        artifact_path = (args.artifact or paths.artifacts / "final_match_model.joblib").resolve()
        report_path = (args.report or paths.reports / "final_model_selection.json").resolve()
        match_features = pd.read_parquet(match_path)
        model_config = load_model_config(config_path)
        with evaluation_path.open(encoding="utf-8") as handle:
            evaluation_report = json.load(handle)
        artifact = train_final_match_model(match_features, model_config)
        report = build_final_model_report(artifact, evaluation_report)
        write_final_model_outputs(artifact, report, artifact_path, report_path)
        print("MPL final model selection and training")
        print(f"Selected model: {artifact['model_name']}")
        print(
            "Training seasons / matches: "
            f"S{artifact['training_season_min']}-S{artifact['training_season_max']} / "
            f"{artifact['training_match_count']}"
        )
        print(f"Calibration observations: {artifact['calibration_observation_count']}")
        print(f"Artifact: {artifact_path}")
        print(f"Report: {report_path}")
        return 0

    if args.command == "update-season18-predictions":
        paths = get_project_paths()
        canonical_dir = (args.canonical_dir or paths.processed / "canonical").resolve()
        season18_dir = (args.season18_dir or paths.data / "season18").resolve()
        feature_config_path = (
            args.feature_config or paths.root / "config" / "feature_config.json"
        ).resolve()
        artifact_path = (
            args.model_artifact or paths.artifacts / "final_match_model.joblib"
        ).resolve()
        config_path = (args.config or paths.root / "config" / "simulation_config.json").resolve()
        prediction_dir = (args.prediction_dir or paths.processed / "predictions").resolve()
        report_path = (args.report or paths.reports / "season18_prediction_updates.json").resolve()
        latest_report_path = (
            args.latest_report or paths.reports / "season18_simulation_report.json"
        ).resolve()
        tables = load_canonical_tables(canonical_dir)
        teams, rosters, schedule = load_season18_tables(season18_dir)
        feature_config = load_feature_config(feature_config_path)
        artifact = load_final_match_model(artifact_path)
        simulation_config = load_simulation_config(config_path)
        if args.iterations is not None:
            if args.iterations <= 0:
                parser.error("--iterations must be greater than zero")
            simulation_config["iterations"] = args.iterations
        predictions, match_probabilities, report, latest_report = build_season18_prediction_history(
            tables,
            teams,
            rosters,
            schedule,
            feature_config,
            artifact,
            simulation_config,
        )
        outputs = write_season18_prediction_history(
            predictions,
            match_probabilities,
            report,
            latest_report,
            prediction_dir,
            report_path,
            latest_report_path,
        )
        latest = predictions.loc[
            predictions["snapshot_order"].eq(predictions["snapshot_order"].max())
        ].sort_values("champion_probability", ascending=False)
        leader = latest.iloc[0]
        print("MPL Season 18 preseason reconstruction and weekly updates")
        print(
            f"Snapshots: {report['snapshot_count']} "
            f"(preseason {report['preseason_snapshot_count']}, "
            f"weekly {report['weekly_snapshot_count']})"
        )
        print(f"Latest completed week: {report['latest_completed_week']}")
        print(f"Latest leader: {leader['team_name']} ({leader['champion_probability']:.2%})")
        print(f"Invalid probability sums: {report['validation']['invalid_probability_sum_count']}")
        print(f"Prediction history: {outputs['history']}")
        print(f"Report: {report_path}")
        return int(report["validation"]["invalid_probability_sum_count"] > 0)

    if args.command == "explain-season18":
        paths = get_project_paths()
        artifact_path = (
            args.model_artifact or paths.artifacts / "final_match_model.joblib"
        ).resolve()
        prediction_dir = (args.prediction_dir or paths.processed / "predictions").resolve()
        predictions_path = (
            args.predictions or prediction_dir / "season18_snapshot_predictions.parquet"
        ).resolve()
        match_path = (
            args.match_probabilities
            or prediction_dir / "season18_snapshot_match_probabilities.parquet"
        ).resolve()
        report_path = (args.report or paths.reports / "explainability_report.json").resolve()
        artifact = load_final_match_model(artifact_path)
        predictions = pd.read_parquet(predictions_path)
        match_probabilities = pd.read_parquet(match_path)
        global_importance = build_global_importance(artifact)
        match_explanations = build_match_explanations(artifact, match_probabilities)
        team_explanations = build_team_explanations(predictions, match_probabilities)
        report = build_explainability_report(
            artifact,
            global_importance,
            match_explanations,
            team_explanations,
        )
        outputs = write_explainability_outputs(
            global_importance,
            match_explanations,
            team_explanations,
            report,
            prediction_dir,
            report_path,
        )
        print("MPL Season 18 model explainability")
        print(f"Global transformed features: {len(global_importance)}")
        print(f"Explained upcoming matches: {match_explanations['match_id'].nunique()}")
        print(f"Explained teams: {len(team_explanations)}")
        print(f"Match explanations: {outputs['matches']}")
        print(f"Report: {report_path}")
        return 0

    if args.command == "simulate-season18":
        paths = get_project_paths()
        canonical_dir = (args.canonical_dir or paths.processed / "canonical").resolve()
        season18_dir = (args.season18_dir or paths.data / "season18").resolve()
        feature_config_path = (
            args.feature_config or paths.root / "config" / "feature_config.json"
        ).resolve()
        artifact_path = (
            args.model_artifact or paths.artifacts / "final_match_model.joblib"
        ).resolve()
        config_path = (args.config or paths.root / "config" / "simulation_config.json").resolve()
        simulation_path = (
            args.simulation_output
            or paths.processed / "predictions" / "season18_simulation.parquet"
        ).resolve()
        match_output_path = (
            args.match_output
            or paths.processed / "predictions" / "season18_match_probabilities.parquet"
        ).resolve()
        report_path = (args.report or paths.reports / "season18_simulation_report.json").resolve()
        tables = load_canonical_tables(canonical_dir)
        teams = pd.read_csv(season18_dir / "teams.csv")
        schedule = pd.read_csv(season18_dir / "schedule_results.csv")
        schedule["scheduled_at"] = pd.to_datetime(schedule["scheduled_at"], utc=True).dt.tz_convert(
            "Asia/Jakarta"
        )
        for column in ("team_a_score", "team_b_score"):
            schedule[column] = pd.to_numeric(schedule[column], errors="coerce").astype("Int64")
        feature_config = load_feature_config(feature_config_path)
        artifact = load_final_match_model(artifact_path)
        simulation_config = load_simulation_config(config_path)
        if args.iterations is not None:
            if args.iterations <= 0:
                parser.error("--iterations must be greater than zero")
            simulation_config["iterations"] = args.iterations
        match_probabilities, tracker, history = build_season18_match_probabilities(
            tables, schedule, teams, feature_config, artifact
        )
        simulation, report = simulate_season18(
            tables,
            schedule,
            teams,
            match_probabilities,
            tracker,
            history,
            artifact,
            simulation_config,
        )
        write_simulation_outputs(
            simulation,
            match_probabilities,
            report,
            simulation_path,
            match_output_path,
            report_path,
        )
        leader = simulation.iloc[0]
        print("MPL Indonesia Season 18 Monte Carlo simulation")
        print(f"Iterations: {report['iterations']}")
        print(
            "Completed / simulated regular matches: "
            f"{report['input_scope']['completed_match_count']} / "
            f"{report['input_scope']['simulated_regular_match_count']}"
        )
        print(
            f"Highest champion probability: {leader['team_name']} "
            f"({leader['champion_probability']:.2%})"
        )
        print(f"Blocking issues: {report['validation']['blocking_issue_count']}")
        print(f"Simulation: {simulation_path}")
        print(f"Match probabilities: {match_output_path}")
        print(f"Report: {report_path}")
        return int(report["validation"]["blocking_issue_count"] > 0)

    parser.error(f"Unknown command: {args.command}")
    return 2
