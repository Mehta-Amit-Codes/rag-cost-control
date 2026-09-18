import json
import os
import time
from dataclasses import dataclass

import numpy as np
import redis

redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/1"))

SIMILARITY_THRESHOLD = float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.95"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(60 * 60 * 6)))  # 6h
MAX_ENTRIES_SCANNED = 500  # cap the linear scan cost per lookup


@dataclass
class CacheEntry:
    query: str
    answer: str
    embedding: list[float]
    created_at: float
    model_used: str
    cost_usd: float


def _key(namespace: str) -> str:
    return f"semcache:{namespace}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def lookup(namespace: str, query_embedding: list[float]) -> tuple[CacheEntry | None, float]:
    """Returns (best_matching_entry_or_None, best_similarity_score)."""
    raw_entries = redis_client.lrange(_key(namespace), 0, MAX_ENTRIES_SCANNED - 1)
    if not raw_entries:
        return None, 0.0

    q_vec = np.array(query_embedding)
    best_entry, best_score = None, 0.0

    for raw in raw_entries:
        record = json.loads(raw)
        if time.time() - record["created_at"] > CACHE_TTL_SECONDS:
            continue  # stale, ignore (lazy expiry -- see evict_stale below for active cleanup)
        cached_vec = np.array(record["embedding"])
        score = _cosine(q_vec, cached_vec)
        if score > best_score:
            best_score = score
            best_entry = record

    if best_entry and best_score >= SIMILARITY_THRESHOLD:
        return CacheEntry(**best_entry), best_score
    return None, best_score


def store(namespace: str, query: str, embedding: list[float], answer: str,
          model_used: str, cost_usd: float) -> None:
    entry = CacheEntry(
        query=query, answer=answer, embedding=embedding,
        created_at=time.time(), model_used=model_used, cost_usd=cost_usd,
    )
    redis_client.lpush(_key(namespace), json.dumps(entry.__dict__))
    redis_client.ltrim(_key(namespace), 0, MAX_ENTRIES_SCANNED - 1)  # bound list size
    redis_client.expire(_key(namespace), CACHE_TTL_SECONDS)


def evict_stale(namespace: str) -> int:
    """Active cleanup, e.g. run periodically via a scheduled job. Returns count removed."""
    raw_entries = redis_client.lrange(_key(namespace), 0, -1)
    fresh = [r for r in raw_entries if time.time() - json.loads(r)["created_at"] <= CACHE_TTL_SECONDS]
    removed = len(raw_entries) - len(fresh)
    if removed:
        pipe = redis_client.pipeline()
        pipe.delete(_key(namespace))
        if fresh:
            pipe.rpush(_key(namespace), *fresh)
        pipe.execute()
    return removed
