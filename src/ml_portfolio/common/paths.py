"""Path management for ML Portfolio projects."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def get_repo_root() -> Path:
    """Find the repository root by looking for pyproject.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current directory
    return current


@dataclass
class ProjectPaths:
    """Paths for a specific project."""

    project_name: str
    repo_root: Path

    @property
    def project_dir(self) -> Path:
        """Root directory of the project."""
        return self.repo_root / "projects" / self.project_name

    @property
    def configs_dir(self) -> Path:
        """Configuration files directory."""
        return self.project_dir / "configs"

    @property
    def scripts_dir(self) -> Path:
        """Scripts directory."""
        return self.project_dir / "scripts"

    @property
    def notebooks_dir(self) -> Path:
        """Notebooks directory."""
        return self.project_dir / "notebooks"

    @property
    def project_module_dir(self) -> Path:
        """Project-specific Python module directory."""
        return self.project_dir / "project"

    @property
    def tests_dir(self) -> Path:
        """Tests directory."""
        return self.project_dir / "tests"

    @property
    def data_dir(self) -> Path:
        """Data directory for this project."""
        path = self.repo_root / "data" / self.project_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def artifacts_dir(self) -> Path:
        """Artifacts directory for this project."""
        path = self.repo_root / "artifacts" / self.project_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reports_dir(self) -> Path:
        """Reports directory for this project."""
        path = self.repo_root / "reports" / self.project_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_report_dir(self, run_id: str) -> Path:
        """Get report directory for a specific run."""
        path = self.reports_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def default_config(self) -> Path:
        """Path to default config file."""
        return self.configs_dir / "default.yaml"


@lru_cache
def get_project_paths(project_name: str) -> ProjectPaths:
    """Get cached project paths instance."""
    return ProjectPaths(project_name=project_name, repo_root=get_repo_root())
