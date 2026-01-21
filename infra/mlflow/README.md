# MLflow Server

Local MLflow tracking server for experiment management.

## Quick Start

```bash
# Start server
docker compose up -d

# View logs
docker compose logs -f

# Stop server
docker compose down
```

## Access

- **UI**: http://localhost:5000
- **API**: http://localhost:5000/api/2.0/mlflow

## Configuration

The server uses:
- **Backend Store**: SQLite (persistent, stored in volume)
- **Artifact Store**: Local filesystem (mounted volume)

## Data Persistence

Data is stored in Docker volumes and mounted directories:
- `mlflow-data` volume: SQLite database
- `./mlruns`: Run metadata
- `./mlartifacts`: Model artifacts

To reset all data:
```bash
docker compose down -v
rm -rf mlruns mlartifacts
```

## Environment Variables

Set these in your `.env` file:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Production Considerations

For production deployments, consider:
1. PostgreSQL backend instead of SQLite
2. S3/GCS for artifact storage
3. Authentication (MLflow doesn't provide built-in auth)
4. Reverse proxy with TLS

Example production docker-compose would use:
```yaml
environment:
  - MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@host/mlflow
  - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://bucket/mlflow
```
