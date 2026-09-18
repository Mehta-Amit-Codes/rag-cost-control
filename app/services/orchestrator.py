import os
import time
import requests
from sentence_transformers import SentenceTransformer

from app.core.observability import RequestEvent, log_event
from app.services import semantic_cache
from app.services.model_router import (
    RouteDecision, classify_complexity, estimate_cost_usd, needs_escalation,
)

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Project 1's API base URL + this tenant's API key, used to fetch context
# when the caller doesn't already have retrieved chunks in hand.
RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def _fetch_context_from_project1(api_key: str, question: str, top_k: int = 5) -> str:
    resp = requests.post(
        f"{RAG_API_URL}/query/retrieve",
        headers={"X-API-Key": api_key},
        json={"question": question, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    sources = resp.json()["sources"]
    return "\n\n".join(f"[{s['chunk_index']}] {s['text']}" for s in sources)


def smart_query(namespace: str, query: str, context: str = "", api_key: str = "") -> dict:
    start = time.perf_counter()

    if not context and api_key:
        context = _fetch_context_from_project1(api_key, query)

    query_embedding = _embedder.encode(query, normalize_embeddings=True).tolist()

    # 1. Semantic cache check
    cached, similarity = semantic_cache.lookup(namespace, query_embedding)
    if cached:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_event(RequestEvent(
            namespace=namespace, query=query, cache_hit=True, cache_similarity=similarity,
            model_used=cached.model_used, route_decision="cache", escalated=False,
            tokens_in=0, tokens_out=0, cost_usd=0.0, latency_ms=latency_ms, timestamp=time.time(),
        ))
        return {
            "answer": cached.answer,
            "cache_hit": True,
            "cache_similarity": round(similarity, 4),
            "model_used": cached.model_used,
            "cost_usd": 0.0,
            "latency_ms": latency_ms,
        }

    # 2. Route: cheap vs strong model, up front
    routing = classify_complexity(query)
    answer, tokens_in, tokens_out = _call_llm(routing.model, query, context)
    escalated = False

    # 3. Post-hoc escalation if the cheap model looks unsure
    if routing.decision == RouteDecision.CHEAP and needs_escalation(answer):
        from app.services.model_router import STRONG_MODEL
        answer, tokens_in2, tokens_out2 = _call_llm(STRONG_MODEL, query, context)
        tokens_in += tokens_in2
        tokens_out += tokens_out2
        routing.model = STRONG_MODEL
        escalated = True

    cost_usd = estimate_cost_usd(routing.model, tokens_in, tokens_out)
    latency_ms = int((time.perf_counter() - start) * 1000)

    # 4. Log + cache the fresh answer
    log_event(RequestEvent(
        namespace=namespace, query=query, cache_hit=False, cache_similarity=similarity,
        model_used=routing.model, route_decision=routing.reason, escalated=escalated,
        tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd,
        latency_ms=latency_ms, timestamp=time.time(),
    ))
    semantic_cache.store(namespace, query, query_embedding, answer, routing.model, cost_usd)

    return {
        "answer": answer,
        "cache_hit": False,
        "cache_similarity": round(similarity, 4),
        "model_used": routing.model,
        "route_reason": routing.reason,
        "escalated": escalated,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 6),
        "latency_ms": latency_ms,
    }


def _call_llm(model: str, query: str, context: str) -> tuple[str, int, int]:
    """Grok (xAI) exposes an OpenAI-compatible API -- the official `openai`
    client works unmodified, pointed at xAI's base_url."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GROK_API_KEY"],
        base_url=os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1"),
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}" if context else query
    response = client.chat.completions.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    return text, response.usage.prompt_tokens, response.usage.completion_tokens