# Deployment

Status: **local Docker files written and validated with `docker compose config`; image build not run here (Docker Hub
blocked); no cloud deployment performed.** CI builds and smoke-tests the image on GitHub.

## Local
```bash
docker compose up --build            # API http://localhost:8000/docs, dashboard http://localhost:8501
docker compose --profile llm up -d   # + Ollama; then: docker compose exec ollama ollama pull llama3.2:3b
                                     #   and set OLLAMA_URL=http://ollama:11434 for the api service
```
The image builds the demo pipeline at build time (synthetic data), runs as a non-root user, and exposes a health check.

## AWS (guide)
Option A - **App Runner** (simplest): one service for the API, one for the dashboard.
1. `aws ecr create-repository --repository-name gameops-ai`
2. `docker build -t gameops-ai . && docker tag gameops-ai <acct>.dkr.ecr.<region>.amazonaws.com/gameops-ai:1.0.0 && docker push …`
3. Create an App Runner service from the ECR image, port 8000, health check `/health`, 1 vCPU / 2 GB.
4. A second service with start command `streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0`.

Option B - **ECS Fargate**: one task definition with two containers (api, dashboard) behind an Application Load
Balancer (path routing `/api*` → api). For an LLM, run Ollama on an EC2 GPU instance (e.g. g5.xlarge) in the same VPC
and set `OLLAMA_URL`; App Runner/Fargate have no GPUs.

Configuration: all settings are environment variables (`.env.example`). No secrets are needed for the demo; for real
data store collected reviews in S3 or EFS, never in the image.

## Scaling notes (measured locally, 2 vCPU)
~32 req/s at p95 57 ms with realistic think time; ~114 req/s at saturation (p95 540 ms). Read-only DuckDB and FAISS are
in-process, so the service scales horizontally by adding replicas (each loads its own read-only copy). Beyond a single
machine's corpus, move to a managed vector store and a warehouse; the agent's latency would then be dominated by the LLM.
