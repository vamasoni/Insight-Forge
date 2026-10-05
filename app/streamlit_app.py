"""Streamlit UI.   streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insightforge import runtime  # noqa: E402
from insightforge.evidence import CalcEvidence, QueryEvidence  # noqa: E402

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("InsightForge")
st.caption("Ask a question about the data. Every finding links to the query or test behind it.")

EXAMPLES = ["Do late deliveries get worse review scores?",
            "Which product categories bring the most revenue, and how do their review scores compare?",
            "Is there a relationship between payment installments and order value?",
            "How has monthly order volume changed, and were there unusual months?"]

with st.sidebar:
    s = runtime.settings()
    st.markdown(f"**Database:** `{s.db_path}`  \n**Model:** `{s.provider_label}`")
    ex = st.radio("Examples", EXAMPLES, index=None)

q = st.text_input("Question", value=ex or "")
if st.button("Analyze", type="primary", disabled=not q.strip()):
    with st.spinner("Planning, querying, testing, checking claims..."):
        st.session_state["report"] = runtime.agent().ask(q.strip())

rep = st.session_state.get("report")
if rep:
    m = rep.metrics()
    c = st.columns(4)
    c[0].metric("Supported claims", m["claims_supported"])
    c[1].metric("Removed by critic", m["claims_rejected"])
    c[2].metric("Cost", f"${m['cost_usd']:.4f}")
    c[3].metric("Latency", f"{m['latency_s']}s")
    st.subheader("Answer")
    st.write(rep.summary)
    for claim in rep.claims:
        with st.expander(f"{claim.id}: {claim.text}"):
            for eid in claim.evidence_ids:
                ev = rep.store.get(eid)
                if isinstance(ev, QueryEvidence):
                    st.markdown(f"**{eid}** · {ev.row_count:,} rows · {ev.attempts} attempt(s)")
                    st.code(ev.sql, language="sql")
                    if eid in rep.store.frames:
                        st.dataframe(rep.store.frames[eid].head(50), use_container_width=True)
                elif isinstance(ev, CalcEvidence):
                    st.markdown(f"**{eid}** · {ev.method} (inputs: {', '.join(ev.inputs)})")
                    st.json({k: v for k, v in ev.outputs.items() if k not in ("table", "anomalies")})
    for v in rep.store.charts():
        st.plotly_chart(pio.from_json(v.figure_json), use_container_width=True, key=v.id)
        st.caption(f"{v.id}, built from {', '.join(v.inputs)}")
    if rep.caveats or rep.failures:
        st.subheader("Caveats")
        for cv in rep.caveats:
            st.write(f"- {cv}")
        for f in rep.failures:
            st.write(f"- Could not answer \"{f['question']}\": {f['error']}")
    if rep.rejected:
        st.subheader("Removed by the critic")
        for r in rep.rejected:
            st.write(f"- ~~{r.text}~~ ({r.status}: {'; '.join(r.critic_notes)})")
    st.download_button("Download report (Markdown)", rep.to_markdown(), file_name=f"{rep.run_id}.md")
    st.download_button("Download evidence (JSON)", json.dumps(rep.to_dict(), default=str, indent=2),
                       file_name=f"{rep.run_id}.json")
