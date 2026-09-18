import os
import re
from dataclasses import dataclass
from enum import Enum

CHEAP_MODEL = os.environ.get("CHEAP_MODEL", "grok-4-fast")
STRONG_MODEL = os.environ.get("STRONG_MODEL", "grok-4")

# Rough $/1K tokens for cost estimation/reporting (illustrative, update to
# current published pricing before using this for real budgeting).
COST_PER_1K_TOKENS = {
    CHEAP_MODEL: {"input": 0.001, "output": 0.005},
    STRONG_MODEL: {"input": 0.003, "output": 0.015},
}

LOW_CONFIDENCE_MARKERS = re.compile(
    r"\b(i'?m not sure|i don'?t know|unclear|cannot determine|insufficient (context|information)|"
    r"it'?s hard to say|may not be accurate)\b",
    re.IGNORECASE,
)


class RouteDecision(Enum):
    CHEAP = "cheap"
    STRONG = "strong"


@dataclass
class RoutingResult:
    model: str
    decision: RouteDecision
    reason: str


def classify_complexity(query: str) -> RoutingResult:
    """
    Up-front heuristic routing. Signals for "needs the strong model":
    - long / multi-part questions (multiple '?' or 'and')
    - requests for reasoning, comparison, computation, synthesis
    - very short queries route cheap (likely simple lookups)
    """
    q = query.strip().lower()
    word_count = len(q.split())

    reasoning_markers = ["compare", "why", "explain", "analyze", "trade-off",
                          "pros and cons", "summarize across", "calculate", "how does"]
    multi_part = q.count("?") > 1 or " and " in q

    if word_count <= 8 and not any(m in q for m in reasoning_markers):
        return RoutingResult(CHEAP_MODEL, RouteDecision.CHEAP, "short, simple query")

    if multi_part or any(m in q for m in reasoning_markers) or word_count > 40:
        return RoutingResult(STRONG_MODEL, RouteDecision.STRONG, "multi-part or reasoning-heavy query")

    return RoutingResult(CHEAP_MODEL, RouteDecision.CHEAP, "default: no strong-model signal")


def needs_escalation(cheap_model_answer: str) -> bool:
    """Post-hoc check: does the cheap model's own answer signal low confidence?"""
    return bool(LOW_CONFIDENCE_MARKERS.search(cheap_model_answer))


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return (tokens_in / 1000) * rates["input"] + (tokens_out / 1000) * rates["output"]
