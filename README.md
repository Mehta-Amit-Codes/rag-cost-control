# RAG Cost Control Layer

[![Diagram](https://img.shields.io/badge/gitdiagram-view%20architecture-blue)](https://gitdiagram.com/Mehta-Amit-Codes/rag-cost-control)

Reference implementation of the "Production RAG with Caching & Cost
Controls" blueprint: semantic caching, tiered (BM25 + vector) retrieval,
model routing/cascade with confidence-based escalation, and a cost/latency
observability layer.

Designed to sit **in front of** a RAG backend such as Project 1
(Multi-Tenant RAG-as-a-Service) — pass its tenant_id in as `namespace` to
get per-tenant cost tracking that lines up with Project 1's own usage
metering.

## How the pieces fit together

```
query --> [semantic cache check] --hit--> return cached answer, $0
              |
             miss
              v
        [complexity router] --> cheap model --> low confidence? --escalate--> strong model
              |                                        |
              v                                        v
        [observability log: cost, latency, cache outcome, escalation]
              |
              v
        [store answer in semantic cache for next near-duplicate query]
```

## Run it

```bash
cp .env.example .env
# then edit .env and fill in your real GROK_API_KEY

docker compose up -d redis
pip install -r requirements.txt

uvicorn app.main:app --port 8001 --reload
```

`.env` is loaded automatically (via `python-dotenv`) by `main.py` and `streamlit_app.py` — no need to `export` variables manually in each terminal. It's git-ignored, so never commit it; `.env.example` documents every variable the app reads.

In a second terminal:
```bash
export COST_CONTROL_API_URL=http://localhost:8001
streamlit run streamlit_app.py
```

## Demo flow

1. In the Streamlit **Try it** tab, ask: *"What is the refund policy?"*
   with some context pasted in. It misses the cache, routes to the cheap
   model (short factual question), and logs cost + latency.
2. Ask a near-duplicate: *"How do refunds work?"* — this should hit the
   semantic cache (similarity shown, cost $0.00).
3. Ask something clearly multi-part/reasoning-heavy: *"Compare the refund
   policy to the warranty policy and explain which is more generous and
   why"* — this routes straight to the strong model.
4. Check the **Dashboard** tab for aggregate cache hit rate, total cost,
   avg latency, and escalation count over the selected window.

## API

- `POST /smart-query` — `{namespace, query, context}` → cached or freshly
  generated answer, with cost/latency/routing metadata.
- `GET /stats/{namespace}?window_seconds=86400` — aggregate metrics for
  the dashboard.

## Integration with Project 1

Two changes wire this project to Project 1:

**1. `multi-tenant-rag/app/routers/query.py` (Project 1)** — adds a
retrieval-only `POST /query/retrieve` endpoint that returns chunks
without generating an answer, so Project 1 doesn't also spend an LLM call
on top of this project's routed one.

**2. `rag-cost-control/app/services/orchestrator.py` (this project)** —
`smart_query()` takes an optional `api_key` (Project 1's tenant API key).
If `context` isn't supplied directly, it's fetched automatically by
calling Project 1's new `/query/retrieve` endpoint.

Call it like:

```bash
curl -X POST localhost:8001/smart-query \
  -H "Content-Type: application/json" \
  -d '{
        "namespace": "<project1_tenant_id>",
        "query": "What is the refund policy?",
        "api_key": "<project1_tenant_api_key>"
      }'
```

Use Project 1's `tenant.id` as this project's `namespace` so cache
entries, cost, and escalation stats are tracked per tenant — you can
literally paste Project 1's `usage_events` cost figures next to this
project's `/stats/{namespace}` response to show the before/after savings
from caching + routing.

## Notes on the illustrative pieces

- `COST_PER_1K_TOKENS` in `model_router.py` uses placeholder rates —
  update to current published pricing before treating cost figures as
  real budget numbers.
- The semantic cache does a linear similarity scan (fine to a few
  thousand entries/namespace). For larger scale, swap in `hnswlib` or a
  vector DB with per-entry TTL.
- `tiered_retrieval.py` (BM25 pre-filter) is provided standalone — wire it
  in front of Project 1's `retrieve_chunks` for large corpora by first
  calling `BM25PreFilter.prefilter()` to get candidate chunk IDs, then
  restricting the pgvector query to that ID set.

## Project layout

```
app/
  core/
    observability.py    # SQLite request log: cost, latency, cache hit, escalation
  services/
    semantic_cache.py     # Redis-backed near-duplicate query cache
    model_router.py         # cheap/strong routing + confidence-based escalation
    tiered_retrieval.py       # BM25 pre-filter before vector search
    orchestrator.py            # ties cache + routing + logging together
  main.py                       # FastAPI: /smart-query, /stats/{namespace}
streamlit_app.py                # dashboard: try-it tester + cost/cache metrics
```