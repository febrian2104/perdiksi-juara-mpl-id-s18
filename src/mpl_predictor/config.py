from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Filesystem locations used by the project."""

    root: Path
    data: Path
    interim: Path
    processed: Path
    artifacts: Path
    reports: Path
    figures: Path


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    """Return project paths, resolving from the installed source tree by default."""
    project_root = (root or Path(__file__).resolve().parents[2]).resolve()
    return ProjectPaths(
        root=project_root,
        data=project_root / "data",
        interim=project_root / "data" / "interim",
        processed=project_root / "data" / "processed",
        artifacts=project_root / "artifacts",
        reports=project_root / "reports",
        figures=project_root / "reports" / "figures",
    )
