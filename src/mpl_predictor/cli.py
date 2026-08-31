import argparse
import json
from collections.abc import Sequence
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
from mpl_predictor.data.semantic_audit import (
    SemanticAuditReport,
    audit_semantics,
    write_semantic_report,
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

    parser.error(f"Unknown command: {args.command}")
    return 2
