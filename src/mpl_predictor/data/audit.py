import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mpl_predictor.data.catalog import DatasetFile, discover_dataset_files
from mpl_predictor.data.contracts import HISTORICAL_SEASONS, REQUIRED_COLUMNS, expected_tables

_TRUE_VALUES = {"1", "true", "yes", "y"}


@dataclass(frozen=True, slots=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class AuditReport:
    files: tuple[DatasetFile, ...]
    row_counts: dict[tuple[int, str], int]
    issues: tuple[AuditIssue, ...]

    @property
    def errors(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def total_rows(self) -> int:
        return sum(self.row_counts.values())

    def summary_rows(self) -> list[dict[str, int | str]]:
        return [
            {"season": season, "table": table, "rows": row_count}
            for (season, table), row_count in sorted(self.row_counts.items())
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_count": len(self.files),
            "total_rows": self.total_rows,
            "row_counts": self.summary_rows(),
            "issues": [asdict(issue) for issue in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        return header, list(reader)


def _inspect_file(dataset_file: DatasetFile) -> tuple[int, list[AuditIssue]]:
    issues: list[AuditIssue] = []
    try:
        header, rows = _read_csv(dataset_file.path)
    except (OSError, UnicodeError, csv.Error) as exc:
        return 0, [
            AuditIssue(
                severity="error",
                code="unreadable_csv",
                message=str(exc),
                path=str(dataset_file.path),
            )
        ]

    required = REQUIRED_COLUMNS.get(dataset_file.table)
    if required is None:
        issues.append(
            AuditIssue(
                severity="warning",
                code="unknown_table",
                message=f"No data contract is defined for table '{dataset_file.table}'.",
                path=str(dataset_file.path),
            )
        )
    else:
        missing_columns = sorted(required - set(header))
        if missing_columns:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="missing_columns",
                    message=f"Missing required columns: {', '.join(missing_columns)}.",
                    path=str(dataset_file.path),
                )
            )

    if not rows:
        issues.append(
            AuditIssue(
                severity="error",
                code="empty_file",
                message="CSV contains no data rows.",
                path=str(dataset_file.path),
            )
        )

    malformed_rows = sum(None in row for row in rows)
    if malformed_rows:
        issues.append(
            AuditIssue(
                severity="error",
                code="malformed_rows",
                message=f"Found {malformed_rows} rows with more values than header columns.",
                path=str(dataset_file.path),
            )
        )

    if dataset_file.table == "matches":
        match_ids = [row.get("match_id", "").strip() for row in rows]
        duplicate_count = len(match_ids) - len(set(match_ids))
        if duplicate_count:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="duplicate_match_id",
                    message=f"Found {duplicate_count} duplicate match IDs.",
                    path=str(dataset_file.path),
                )
            )

    if dataset_file.table == "championships":
        champion_count = sum(
            row.get("champion", "").strip().lower() in _TRUE_VALUES for row in rows
        )
        if champion_count != 1:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="invalid_champion_count",
                    message=f"Expected one champion row, found {champion_count}.",
                    path=str(dataset_file.path),
                )
            )

    return len(rows), issues


def audit_data(data_dir: Path) -> AuditReport:
    """Audit file presence, schema, row shape, match IDs, and champion labels."""
    files = discover_dataset_files(data_dir)
    issues: list[AuditIssue] = []
    row_counts: dict[tuple[int, str], int] = {}

    if not files:
        issues.append(
            AuditIssue(
                severity="error",
                code="no_dataset_files",
                message=f"No MPL season CSV files found below {data_dir}.",
                path=str(data_dir),
            )
        )
        return AuditReport(files=(), row_counts={}, issues=tuple(issues))

    files_by_key: dict[tuple[int, str], list[DatasetFile]] = defaultdict(list)
    for dataset_file in files:
        files_by_key[(dataset_file.season, dataset_file.table)].append(dataset_file)

    seasons_to_check = sorted(set(HISTORICAL_SEASONS) | {item.season for item in files})
    for season in seasons_to_check:
        for table in sorted(expected_tables(season)):
            candidates = files_by_key.get((season, table), [])
            if not candidates:
                issues.append(
                    AuditIssue(
                        severity="error",
                        code="missing_table",
                        message=f"Season {season} is missing required table '{table}'.",
                    )
                )
            elif len(candidates) > 1:
                issues.append(
                    AuditIssue(
                        severity="error",
                        code="duplicate_table_file",
                        message=f"Season {season} has {len(candidates)} files for table '{table}'.",
                    )
                )

    for dataset_file in files:
        row_count, file_issues = _inspect_file(dataset_file)
        row_counts[(dataset_file.season, dataset_file.table)] = row_count
        issues.extend(file_issues)

    issues.sort(key=lambda issue: (issue.severity, issue.code, issue.path or ""))
    return AuditReport(files=tuple(files), row_counts=row_counts, issues=tuple(issues))
