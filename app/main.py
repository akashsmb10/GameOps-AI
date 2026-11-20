"""GameOps AI - FastAPI application.

Run:  uvicorn app.main:app --port 8000      (docs at http://localhost:8000/docs)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.services.state import load_services, services
from src.analytics.store import InvalidQuery
from src.monitoring import telemetry

VERSION = "1.0.0"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "logger": record.name,
                "msg": record.getMessage()}
        base.update(getattr(record, "extra_fields", {}))
        return json.dumps(base)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log = logging.getLogger("gameops.api")
log.handlers, log.propagate = [handler], False
log.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_services()  # models, indexes and the read-only database are loaded once at startup
    yield


app = FastAPI(title="GameOps AI", version=VERSION, lifespan=lifespan,
              description="Agentic player-feedback intelligence API. Demo data is SYNTHETIC (fictional games).")


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"extra_fields": {"request_id": rid, "path": request.url.path}})
        response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["x-request-id"] = rid
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    telemetry.record_request(request.method, path, response.status_code, ms)
    log.info("request", extra={"extra_fields": {"request_id": rid, "method": request.method, "path": path,
                                                "status": response.status_code, "latency_ms": ms}})
    return response


@app.exception_handler(InvalidQuery)
async def invalid_query(_: Request, exc: InvalidQuery):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(FileNotFoundError)
async def not_built(_: Request, exc: FileNotFoundError):
    return JSONResponse({"detail": f"Service not ready: {exc}"}, status_code=503)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/ready", tags=["ops"])
def ready():
    comps = services().status()
    ok = all(v.get("loaded") for k, v in comps.items() if k != "llm")
    return JSONResponse({"ready": ok, "components": comps}, status_code=200 if ok else 503)


app.include_router(router, prefix="/api/v1")
