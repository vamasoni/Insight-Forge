"""A scripted LLM that plays every role well enough to exercise the full pipeline offline."""
from __future__ import annotations

import json
import re

from insightforge.llm import FakeProvider

PLAN = {"sub_questions": [
    {"id": "S1", "question": "How many delivered orders were late vs on time?", "chart": "bar"},
    {"id": "S2", "question": "Review score and delivery status for each delivered order", "needs_analysis": True,
     "analysis": "t_test", "analysis_args": {"value": "review_score", "group": "delivery_status"}},
]}

SQL = {
    "late vs on time": """```sql
SELECT CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'late' ELSE 'on_time' END
       AS delivery_status, COUNT(*) AS n_orders
FROM orders o WHERE o.order_status = 'delivered' GROUP BY 1 ORDER BY 2 DESC
```""",
    "review score and delivery": """```sql
SELECT o.order_id, r.review_score,
       CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'late' ELSE 'on_time' END
       AS delivery_status
FROM orders o JOIN reviews r ON r.order_id = o.order_id WHERE o.order_status = 'delivered'
```""",
}


def make_fake_llm(bad_first_sql: bool = False, causal_claim: bool = True) -> FakeProvider:
    state = {"sql_calls": 0, "synth_calls": 0}

    def respond(system: str, messages: list[dict], role: str) -> str:
        last = str(messages[-1]["content"])
        if role == "planner":
            return json.dumps(PLAN)
        if role in ("sql", "sql_repair"):
            state["sql_calls"] += 1
            if bad_first_sql and state["sql_calls"] == 1:
                return "```sql\nSELECT delivery_stat, COUNT(*) FROM orders GROUP BY 1\n```"
            q = next(str(m["content"]) for m in messages if m["role"] == "user").lower()
            for key, sql in SQL.items():
                if all(w in q for w in key.split(" ")[:2]):
                    return sql
            return SQL["late vs on time"]
        if role == "synthesizer":
            state["synth_calls"] += 1
            ev = last
            late = re.search(r"late \| (\d+)", ev)
            on_time = re.search(r"on_time \| (\d+)", ev)
            calc = re.search(r"\[(C\d+)\] welch_t_test.*?(\{.*\})", ev, re.S)
            claims = []
            if late and on_time:
                claims.append({"text": f"{late.group(1)} delivered orders arrived late and {on_time.group(1)} "
                                       f"arrived on time.", "evidence_ids": ["Q1"], "kind": "descriptive"})
            if calc:
                out = json.loads(calc.group(2))
                ma, mb = out["mean_a"], out["mean_b"]
                claims.append({"text": f"Mean review score is {ma:.2f} for {out['groups'][0]} orders versus "
                                       f"{mb:.2f} for {out['groups'][1]} orders; the difference is statistically "
                                       f"significant (Welch t-test).", "evidence_ids": [calc.group(1)],
                               "kind": "statistical"})
                if causal_claim and state["synth_calls"] == 1:
                    claims.append({"text": "Late delivery causes customers to leave worse reviews.",
                                   "evidence_ids": [calc.group(1)], "kind": "causal"})
                    claims.append({"text": "About 37% of all orders are late.", "evidence_ids": ["Q1"],
                                   "kind": "descriptive"})
            return json.dumps({"summary": "Late deliveries are associated with lower review scores.",
                               "claims": claims, "caveats": ["Data is observational."]})
        if role == "critic":
            ids = re.findall(r"^(F\d+):", last, re.M)
            return json.dumps({"verdicts": [{"id": i, "verdict": "supported", "reason": "matches"} for i in ids]})
        if role == "judge":
            return json.dumps({"score": 2, "reason": "ok"})
        raise AssertionError(f"unexpected role {role}")

    return FakeProvider(responder=respond)
