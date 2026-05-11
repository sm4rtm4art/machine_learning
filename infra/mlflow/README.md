# MLflow Server

Local MLflow tracking server for experiment management.

## Quick Start

```bash
# Start from repository root
make mlflow

# View logs
make mlflow-logs

# Stop server
make mlflow-stop
```

If you are already inside `infra/mlflow/`, you can still run:

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

After compose changes, restart to apply new server flags:

```bash
make mlflow-stop
make mlflow
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
- `./mlartifacts`: Model artifacts

To reset all data:
```bash
docker compose down -v
rm -rf mlartifacts
```

## Environment Variables

Set these in your `.env` file:
```bash
# Use the running MLflow service
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Prototype Scripts Note

Prototype scripts can use a local SQLite URI (for example `sqlite:///mlflow.db`) when no
server is running. For team usage, prefer the shared local server above so everyone tracks
runs in one place.

## Troubleshooting

### UI is empty

If the MLflow UI has no runs:
1. Confirm the script used `MLFLOW_TRACKING_URI=http://localhost:5000`
2. Restart the server (`make mlflow-stop && make mlflow`)
3. Re-run the prototype script

### Artifact logging errors (`/mlflow` read-only)

This setup uses `--serve-artifacts` so clients upload artifacts through the server.
If you still see artifact errors, restart the stack and check logs:

```bash
make mlflow-logs
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
  - MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI}
  - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://bucket/mlflow
```
