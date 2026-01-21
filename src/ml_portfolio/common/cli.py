"""CLI utilities and main entry point."""

import typer
from rich.console import Console
from rich.table import Table

from ml_portfolio import __version__
from ml_portfolio.common.paths import get_repo_root

app = typer.Typer(
    name="ml-portfolio",
    help="ML Portfolio CLI - Manage machine learning projects",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"ML Portfolio v{__version__}")


@app.command()
def info() -> None:
    """Show repository information."""
    repo_root = get_repo_root()

    table = Table(title="ML Portfolio Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Repository Root", str(repo_root))
    table.add_row("Data Directory", str(repo_root / "data"))
    table.add_row("Artifacts Directory", str(repo_root / "artifacts"))
    table.add_row("Reports Directory", str(repo_root / "reports"))

    console.print(table)


@app.command()
def list_projects() -> None:
    """List all available projects."""
    repo_root = get_repo_root()
    projects_dir = repo_root / "projects"

    if not projects_dir.exists():
        console.print("[yellow]No projects directory found.[/yellow]")
        return

    table = Table(title="Available Projects")
    table.add_column("Project", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Path", style="dim")

    for project_path in sorted(projects_dir.iterdir()):
        if project_path.is_dir() and not project_path.name.startswith("_"):
            # Check for README to determine status
            readme = project_path / "README.md"
            if readme.exists():
                content = readme.read_text()
                if "Coming Soon" in content or "Planned" in content:
                    status = "📋 Planned"
                elif "Active" in content or "🚧" in content:
                    status = "🚧 Active"
                else:
                    status = "✅ Complete"
            else:
                status = "❓ Unknown"

            table.add_row(project_path.name, status, str(project_path.relative_to(repo_root)))

    console.print(table)


if __name__ == "__main__":
    app()
