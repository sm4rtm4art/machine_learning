"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings for ML Portfolio.

    Settings are loaded from environment variables with ML_PORTFOLIO_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ML_PORTFOLIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Model artifacts directory")
    reports_dir: Path = Field(default=Path("reports"), description="Evaluation reports directory")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    mlflow_experiment_name: str = Field(
        default="default",
        description="Default MLflow experiment name",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format",
    )

    # Compute
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="PyTorch device to use",
    )
    seed: int = Field(default=42, description="Global random seed")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
