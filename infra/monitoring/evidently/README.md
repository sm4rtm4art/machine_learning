# Evidently Monitoring

Data and model monitoring using Evidently AI.

## Quick Start

```bash
# Start Evidently service
docker compose up -d

# View logs
docker compose logs -f

# Stop service
docker compose down
```

## Access

- **UI**: http://localhost:8000
- **API**: http://localhost:8000/api

## Features

Evidently provides:
- **Data Drift Detection**: Monitor distribution changes in input features
- **Target Drift**: Track changes in prediction distribution
- **Model Performance**: Monitor accuracy, precision, recall over time
- **Data Quality**: Detect missing values, outliers, schema changes

## Integration

### Python SDK

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=production_df)

# Save to workspace
workspace = RemoteWorkspace("http://localhost:8000")
workspace.add_report(project_id, report)
```

### Creating a Project

```python
from evidently.ui.workspace import RemoteWorkspace

workspace = RemoteWorkspace("http://localhost:8000")
project = workspace.create_project("ocr_pipeline")
```

## Use Cases

### 1. Data Drift Monitoring
Track if production data differs from training data.

### 2. Performance Monitoring
Monitor model metrics over time to detect degradation.

### 3. Batch Evaluation
Compare different model versions or data slices.

## Data Persistence

- `evidently-data` volume: Service data
- `./workspaces`: Project workspaces

## Production Considerations

For production:
1. Use PostgreSQL for persistence
2. Configure authentication
3. Set up alerting for drift detection
4. Integrate with your CI/CD pipeline
