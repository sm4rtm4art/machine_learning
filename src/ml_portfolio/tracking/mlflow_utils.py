"""MLflow tracking utilities."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from ml_portfolio.common.config import get_settings


def setup_mlflow(experiment_name: str | None = None) -> str:
    """Set up MLflow tracking.

    Args:
        experiment_name: Experiment name. Defaults to settings.

    Returns:
        Experiment ID.
    """
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    experiment_name = experiment_name or settings.mlflow_experiment_name
    experiment_id = get_or_create_experiment(experiment_name)

    return experiment_id


def get_or_create_experiment(name: str) -> str:
    """Get or create an MLflow experiment.

    Args:
        name: Experiment name.

    Returns:
        Experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        experiment_id = str(mlflow.create_experiment(name))
    else:
        experiment_id = str(experiment.experiment_id)

    mlflow.set_experiment(name)
    return experiment_id


def log_config(config: DictConfig | dict[str, Any], artifact_path: str = "config") -> None:
    """Log configuration to MLflow.

    Args:
        config: Configuration object (Hydra DictConfig or dict).
        artifact_path: MLflow artifact subdirectory.
    """
    # Convert to dict if needed
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config

    # Log as params (flattened)
    flat_params = _flatten_dict(config_dict)
    # MLflow has 500 char limit for param values
    for key, value in flat_params.items():
        str_value = str(value)
        if len(str_value) > 500:
            str_value = str_value[:497] + "..."
        mlflow.log_param(key, str_value)

    # Log as artifact (full)
    config_path = Path("config.yaml")
    if isinstance(config, DictConfig):
        OmegaConf.save(config, config_path)
    else:
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

    mlflow.log_artifact(str(config_path), artifact_path)
    config_path.unlink()  # Clean up


def log_reproducibility_info(config: DictConfig | dict[str, Any] | None = None) -> None:
    """Log information needed for reproducibility.

    Logs:
    - Git commit hash
    - Git diff (if dirty)
    - Environment hash
    - Timestamp
    - Config hash

    Args:
        config: Optional config to hash.
    """
    tags = {}

    # Git info
    git_commit = _get_git_commit()
    if git_commit:
        tags["git_commit"] = git_commit

    git_dirty = _is_git_dirty()
    tags["git_dirty"] = str(git_dirty)

    # Environment hash
    env_hash = _get_environment_hash()
    tags["environment_hash"] = env_hash

    # Timestamp
    tags["run_timestamp"] = datetime.now(UTC).isoformat()

    # Config hash
    if config is not None:
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        config_str = json.dumps(config_dict, sort_keys=True)
        tags["config_hash"] = hashlib.md5(config_str.encode()).hexdigest()[:8]

    mlflow.set_tags(tags)


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary."""
    items: list[tuple[str, Any]] = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def _get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _is_git_dirty() -> bool:
    """Check if git working directory is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_environment_hash() -> str:
    """Get hash of current Python environment."""
    try:
        result = subprocess.run(
            ["uv", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            packages = result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            packages = ""

    return hashlib.md5(packages.encode()).hexdigest()[:8]
