import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mpl_predictor.config import get_project_paths
from mpl_predictor.data.audit import AuditReport, audit_data
from mpl_predictor.data.normalization import normalize_tables, write_normalized_tables
from mpl_predictor.data.semantic_audit import (
    SemanticAuditReport,
    audit_semantics,
    write_semantic_report,
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

    parser.error(f"Unknown command: {args.command}")
    return 2
