# pysemdist (K8s-ready, S3-enabled)

## Local install & run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pysemdist-api
# or
python -m pysemdist
```

Visit http://localhost:8000/docs for Swagger.

## Env (.env or export)
```env
DB_HOST=your-redshift-cluster.amazonaws.com
DB_PORT=5439
DB_NAME=analytics
DB_USER=service_user
DB_PASSWORD=secret_password

DB_POOL_MIN=1
DB_POOL_MAX=10
DB_CONNECT_TIMEOUT=10
DB_STATEMENT_TIMEOUT_MS=60000

# Outputs & S3
S3_BUCKET=your-bucket-name
S3_PREFIX=pysemdist/exports
S3_UPLOAD=true   # set to true to perform real uploads

# Optional: change output dir
# OUTPUT_DIR=data/zbronze
```

## Endpoints (POST)
- `/extract/country` body:
  ```json
  {"start_date":"2025-01-01","end_date":"2025-12-31","country_code":"US","limit":500}
  ```
- `/extract/city` body:
  ```json
  {"start_date":"2025-01-01","end_date":"2025-12-31","country_code":"US","city":"New York","limit":500}
  ```
- `/extract/dm` body:
  ```json
  {"start_date":"2025-01-01","country_code":"US","dm_id":13056908,"limit":500}
  ```
- `/extract/topic` body:
  ```json
  {"start_date":"2025-01-01","country_code":"US","topic_id":8035,"limit":500}
  ```

Outputs land in `data/zbronze/<group>/<group>_<timestamp>.csv`. If `S3_UPLOAD=true` and `S3_BUCKET` set, files are uploaded with `s3://$S3_BUCKET/$S3_PREFIX/...`.

## Docker
```bash
docker build -t your-registry/pysemdist:0.2.0 .
docker run -p 8000:8000 --env-file .env your-registry/pysemdist:0.2.0
```

## Kubernetes (raw manifests)
```bash
# create secret for DB password
kubectl apply -f k8s/secret.example.yaml
# edit k8s/deployment.yaml for your image and env overrides
kubectl apply -f k8s/deployment.yaml
```

## Helm
```bash
helm upgrade --install pysemdist ./helm/pysemdist       --set image.repository=your-registry/pysemdist       --set image.tag=0.2.0       --set env.DB_HOST=your-redshift-cluster.amazonaws.com       --set env.DB_PORT=5439       --set env.DB_NAME=analytics       --set env.DB_USER=service_user       --set env.S3_BUCKET=your-bucket       --set env.S3_PREFIX=pysemdist/exports       --set env.S3_UPLOAD=true
# Then create the secret for DB password:
# kubectl create secret generic pysemdist-secrets --from-literal=DB_PASSWORD='...'
```

## Cluster Goals (HDBSCAN + Ollama)
Endpoint:
```bash
curl -X POST http://localhost:8000/cluster/goals       -H "Content-Type: application/json"       -d '{"start_date":"2025-01-01","end_date":"2025-12-31","country_code":"US","city":"New York","limit":500}'
```

### Helm: enable Ollama sidecar
In `helm/pysemdist/values.yaml`:
```yaml
ollama:
  enabled: true
  image: ollama/ollama:latest
  model: yi:9b
  pull_on_start: true
```

This will run an Ollama sidecar in the API pod and pre-pull `yi:9b`. The API uses `OLLAMA_HOST=http://localhost:11434` automatically.
