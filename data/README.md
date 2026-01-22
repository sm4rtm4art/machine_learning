# Data Directory

This directory contains datasets for ML portfolio projects.

## Structure

```
data/
├── ocr_pipeline/      # SROIE receipts dataset
├── tabular_boosting/    # Tabular datasets
└── ...                  # Other project data
```

## Downloading Data

Each project has a `download_data.py` script:

```bash
# Example: Download SROIE for OCR project
uv run python projects/ocr_pipeline/scripts/download_data.py
```

## Git Ignore

Data files are excluded from git to keep the repository lightweight:
- CSV, Parquet, and other data formats
- Images in data directories
- Downloaded archives

Only this README and `.gitkeep` files are tracked.

## Data Guidelines

1. **Don't commit data**: Use download scripts for reproducibility
2. **Document sources**: Each project README should specify data sources
3. **Use consistent structure**: Follow `data/<project>/` convention
4. **Verify data**: Use `--verify` flags in download scripts
