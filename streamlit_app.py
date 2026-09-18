"""
Streamlit dashboard for the RAG Cost Control layer.

Two views:
  - Try it: fire queries through the smart-query endpoint and watch
    cache hits / model routing / cost happen in real time.
  - Dashboard: aggregate cost, cache hit rate, latency, and escalation
    rate over a time window -- the cost-vs-quality frontier the blueprint
    asks for.

Run:
    streamlit run streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("COST_CONTROL_API_URL", "http://localhost:8001")

st.set_page_config(page_title="RAG Cost Control", page_icon="💸", layout="wide")
st.title("💸 RAG Cost Control Layer")
st.caption(f"API: {API_BASE_URL} · Semantic cache + model routing + observability")

namespace = st.sidebar.text_input("Namespace (e.g. tenant_id)", value="demo-tenant")
st.sidebar.caption(
    "Namespace scopes both the semantic cache and the stats view — "
    "use your Project 1 tenant_id here to track cost per tenant."
)

tab_try, tab_dashboard = st.tabs(["🧪 Try it", "📊 Dashboard"])

# ---- Try it -------------------------------------------------------------
with tab_try:
    st.subheader("Run a query through the cost-control stack")
    query = st.text_input("Query", placeholder="What is the refund policy?")
    context = st.text_area(
        "Context (optional — paste retrieved chunks from your RAG pipeline)",
        height=120,
        placeholder="Paste retrieved chunk text here, or leave blank for a plain query.",
    )

    if st.button("Run", type="primary") and query:
        with st.spinner("Checking cache, routing, calling model if needed..."):
            resp = requests.post(
                f"{API_BASE_URL}/smart-query",
                json={"namespace": namespace, "query": query, "context": context},
            )
        if resp.ok:
            data = resp.json()

            if data["cache_hit"]:
                st.success(f"⚡ Cache hit (similarity: {data['cache_similarity']:.3f}) — $0.00, no LLM call")
            else:
                badge = "🔺 escalated to strong model" if data.get("escalated") else ""
                st.info(
                    f"🧭 Routed to **{data['model_used']}** {badge} — "
                    f"reason: _{data.get('route_reason', 'n/a')}_"
                )

            st.markdown("### Answer")
            st.write(data["answer"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Cost", f"${data['cost_usd']:.6f}")
            col2.metric("Latency", f"{data['latency_ms']} ms")
            col3.metric("Cache hit", "Yes" if data["cache_hit"] else "No")
        else:
            st.error(f"Failed: {resp.status_code} {resp.text}")

    st.divider()
    st.caption(
        "Try asking the same question two different ways (e.g. \"What's your refund policy?\" "
        "then \"How do refunds work?\") to see the second one hit the semantic cache."
    )

# ---- Dashboard ------------------------------------------------------------
with tab_dashboard:
    st.subheader("Cost & performance, this namespace")

    window_label = st.selectbox("Window", ["Last hour", "Last 24 hours", "Last 7 days"], index=1)
    window_seconds = {"Last hour": 3600, "Last 24 hours": 86400, "Last 7 days": 604800}[window_label]

    resp = requests.get(f"{API_BASE_URL}/stats/{namespace}", params={"window_seconds": window_seconds})
    if resp.ok:
        stats = resp.json()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total requests", stats["total_requests"])
        col2.metric("Cache hit rate", f"{stats['cache_hit_rate']*100:.1f}%")
        col3.metric("Total cost", f"${stats['total_cost_usd']:.4f}")
        col4.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
        col5.metric("Escalations", stats["escalations"])

        st.caption(
            "Cache hit rate and escalation rate are the two levers to watch: "
            "higher cache hits = lower cost with no quality change (same cached answer). "
            "Escalations show how often the cheap model needed a strong-model rescue — "
            "a high rate means the up-front routing threshold should shift toward 'strong'."
        )
    else:
        st.error(f"Failed to load stats: {resp.status_code}")

    st.divider()
    st.caption(
        "For a full request-level view (per-query cost, model, cache outcome), "
        "query the SQLite observability DB directly, or extend `/stats` with a "
        "paginated `/events` endpoint."
    )
