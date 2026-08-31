import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mpl_predictor.config import get_project_paths
from mpl_predictor.data.audit import AuditReport, audit_data


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

    parser.error(f"Unknown command: {args.command}")
    return 2
