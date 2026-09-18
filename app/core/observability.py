import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass

DB_PATH = os.environ.get("OBSERVABILITY_DB", "cost_control_events.db")


@dataclass
class RequestEvent:
    namespace: str
    query: str
    cache_hit: bool
    cache_similarity: float
    model_used: str
    route_decision: str
    escalated: bool
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    timestamp: float


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT,
                query TEXT,
                cache_hit INTEGER,
                cache_similarity REAL,
                model_used TEXT,
                route_decision TEXT,
                escalated INTEGER,
                tokens_in INTEGER,
                tokens_out INTEGER,
                cost_usd REAL,
                latency_ms INTEGER,
                timestamp REAL
            )
        """)


def log_event(event: RequestEvent) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO request_events
               (namespace, query, cache_hit, cache_similarity, model_used, route_decision,
                escalated, tokens_in, tokens_out, cost_usd, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.namespace, event.query, int(event.cache_hit), event.cache_similarity,
             event.model_used, event.route_decision, int(event.escalated),
             event.tokens_in, event.tokens_out, event.cost_usd, event.latency_ms, event.timestamp),
        )


def summary(namespace: str, window_seconds: int = 86400) -> dict:
    since = time.time() - window_seconds
    with _conn() as conn:
        row = conn.execute(
            """SELECT
                   COUNT(*) AS total,
                   SUM(cache_hit) AS cache_hits,
                   SUM(cost_usd) AS total_cost,
                   AVG(latency_ms) AS avg_latency,
                   SUM(escalated) AS escalations
               FROM request_events
               WHERE namespace = ? AND timestamp >= ?""",
            (namespace, since),
        ).fetchone()

    total, cache_hits, total_cost, avg_latency, escalations = row
    total = total or 0
    return {
        "total_requests": total,
        "cache_hits": cache_hits or 0,
        "cache_hit_rate": (cache_hits or 0) / total if total else 0.0,
        "total_cost_usd": total_cost or 0.0,
        "avg_latency_ms": avg_latency or 0.0,
        "escalations": escalations or 0,
    }


def recent_events(namespace: str, limit: int = 50) -> list[dict]:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM request_events WHERE namespace = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (namespace, limit),
        ).fetchall()
    return [dict(r) for r in rows]
