from fastapi import FastAPI

from dotenv import load_dotenv
load_dotenv()  # loads .env for local `uvicorn` runs; no-op under Docker Compose

from pydantic import BaseModel

from app.core.observability import init_db, summary
from app.services.orchestrator import smart_query

app = FastAPI(
    title="RAG Cost Control Layer",
    description="Semantic caching + model routing + cost/latency observability, "
                "designed to sit in front of a RAG backend like Project 1.",
    version="0.1.0",
)

init_db()


class SmartQueryRequest(BaseModel):
    namespace: str  # e.g. tenant_id from Project 1, or any app-defined scope
    query: str
    context: str = ""  # retrieved chunks from your RAG pipeline, if any
    api_key: str = ""  # Project 1 tenant API key; auto-fetches context if `context` is empty


@app.post("/smart-query")
def smart_query_endpoint(req: SmartQueryRequest):
    return smart_query(req.namespace, req.query, req.context, req.api_key)


@app.get("/stats/{namespace}")
def stats(namespace: str, window_seconds: int = 86400):
    return summary(namespace, window_seconds)


@app.get("/health")
def health():
    return {"status": "ok"}