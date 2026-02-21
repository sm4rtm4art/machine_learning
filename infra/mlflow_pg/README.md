

Starts **MLflow** server with **PostgreSQL** and **MinIO**.

Stores database and artifacts in project (non-persistent and git-ignored) for testing.

## Quickstart
Create `.env` by copying the template:
```commandline
cp .env.template .env
```
Use `docker compose` plugin in this directory
```bash
# Start server
docker compose up -d

# View logs
docker compose logs -f

# Stop server
docker compose down
```
Or use shorhands in the repository root:
```bash
# Start server
make mlflow-pg

# View logs
make mlflow-pg-logs
# Stop server
make mlflow-pg-stop
```


### Notes
Docker image versions are fixed in `.env` to:

```bash
POSTGRES_VERSION=15.16
MINIO_RELEASE=RELEASE.2025-08-13T08-35-41Z
MLFLOW_VERSION=v3.10.0
```


Orginal code from

https://github.com/mlflow/mlflow/tree/master/docker-compose

with small adaptations: Store database and artifacts locally (non-persistent) and fix image version. 
