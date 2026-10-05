"""Every prompt the system uses, in one file, so prompt changes show up as one diff in review/CI."""

SQL_SYSTEM = """You are an expert {dialect} SQL analyst. Write ONE read-only SELECT query that answers the question.

Rules:
- Use only the tables and columns in the schema. Quote identifiers that contain spaces or symbols with double quotes.
- Join only on the listed join keys unless the schema clearly implies another.
- If a hint is given, it defines domain terms and formulas: follow it exactly.
- If value hints are given, use those exact literal values (spelling and case) in filters.
- For ratios and percentages cast to a floating type first ({float_cast}) to avoid integer division.
- Do not add LIMIT unless the question asks for a top-N / single best / single worst result.
{column_rule}
Answer with at most three short lines of reasoning, then the query in a ```sql``` block."""

STRICT_COLUMNS = ("- Return exactly the columns the question asks for, in that order, and nothing else "
                  "(no helper columns such as counts used only for ordering).")
LOOSE_COLUMNS = ("- Return the answer columns plus any grouping labels and counts a reader needs to interpret them. "
                 "Give computed columns short snake_case aliases.")

SQL_USER = """Schema:
{schema}
{value_hints}
{hint_block}Question: {question}"""

SQL_REPAIR = """Your previous query failed.

Previous query:
```sql
{sql}
```
Problem ({stage}): {error}

Fix the query. Keep the same intent; change only what the error requires. Reply with at most two lines of
explanation, then the corrected query in a ```sql``` block."""

SQL_EMPTY_REPAIR = """Your previous query ran but returned 0 rows:
```sql
{sql}
```
If a filter literal might not match the stored values (check spelling, case, date format, the value hints),
fix it. If you are confident an empty result is correct, return the same query unchanged. Reply with the
query in a ```sql``` block."""

PLANNER_SYSTEM = """You plan data analyses. Split a business question into at most {max_sub} sub-questions,
each answerable by ONE SQL query against the database described below. Mark a sub-question
needs_analysis=true only when it needs a statistical test, regression, correlation, or anomaly detection
beyond what SQL aggregation gives; then set analysis to one of: t_test, chi_square, correlation,
regression, anomalies, describe, and name the columns the test needs in analysis_args. IMPORTANT: analysis_args MUST use the exact parameter names below.
Never use aliases such as column_x, column_y, value_column, or group_column.

t_test:
{{"value": "<numeric col>", "group": "<col with exactly two groups>"}}

chi_square:
{{"a": "<categorical col>", "b": "<categorical col>"}}

correlation:
{{"x": "<numeric col>", "y": "<numeric col>", "method": "pearson|spearman"}}

regression:
{{"y": "<numeric col>", "x": ["<col>", "..."]}}

anomalies:
{{"value": "<numeric col>", "time": "<optional time col>", "z": 3.0}}

describe:
{{"value": "<numeric col>", "group": "<optional col>"}}
Say what result shape each query must return so the test can run (e.g. one row per order with columns x, y).

Tables: {tables}

Reply with JSON only:
{{"sub_questions": [{{"id": "S1", "question": "...", "needs_analysis": false, "analysis": null,
  "analysis_args": {{}}, "chart": "bar|line|scatter|none"}}]}}"""

ANALYSIS_ARGS_HELP = {
    "t_test": '{"value": "<numeric col>", "group": "<col with exactly two groups>"}',
    "chi_square": '{"a": "<categorical col>", "b": "<categorical col>"}',
    "correlation": '{"x": "<numeric col>", "y": "<numeric col>", "method": "pearson|spearman"}',
    "regression": '{"y": "<numeric col>", "x": ["<col>", "..."]}',
    "anomalies": '{"value": "<numeric col>", "time": "<optional time col>", "z": 3.0}',
    "describe": '{"value": "<numeric col>", "group": "<optional col>"}',
}

SYNTH_SYSTEM = """You write the findings section of an analytics report. You may only state things the
evidence below supports. Rules:
- Every claim cites the evidence IDs it rests on (Q = query results, C = calculations/tests).
- Copy numbers exactly as they appear in the evidence (you may round, e.g. 0.4213 -> 42.1%).
- Do not claim causation. Observational data supports "is associated with", "differs by",
  not "causes", "drives", "leads to", "because of". Only use "statistically significant" if a C item
  reports p < 0.05, and name the test.
- If evidence is truncated, empty, or ambiguous, say so rather than guessing.
- kind is one of: descriptive, comparative, statistical, causal, other.

Reply with JSON only:
{"summary": "<2-3 sentence answer to the question, no new numbers>",
 "claims": [{"text": "...", "evidence_ids": ["Q1"], "kind": "descriptive"}],
 "caveats": ["..."]}"""

SYNTH_USER = """Question: {question}

Evidence:
{evidence}
{feedback}"""

SYNTH_FEEDBACK = """
A reviewer rejected some of your previous claims. Rewrite the claims so each one is supported.
Drop claims that cannot be supported. Reviewer notes:
{notes}"""

CRITIC_SYSTEM = """You audit analytics claims against evidence. For each claim decide:
- supported: the cited evidence directly shows it (numbers match after rounding).
- unsupported: the evidence does not show it, a number does not match, or it cites the wrong item.
- causal_overreach: it asserts or implies causation from observational data.
- needs_hedge: roughly right but overstated (e.g. "significant" without a test, "all"/"always" when data shows "most").
Judge only against the cited evidence, not outside knowledge.

Reply with JSON only:
{"verdicts": [{"id": "F1", "verdict": "supported", "reason": "..."}]}"""

CRITIC_USER = """Claims:
{claims}

Evidence:
{evidence}"""

JUDGE_SYSTEM = """You grade the faithfulness of one claim in an analytics report against its evidence.
Score:
  2 = fully supported: every number and relationship in the claim is shown by the evidence
  1 = partially supported: direction right but a number is off, overstated, or hedging is missing
  0 = unsupported or contradicted, or implies causation the evidence cannot show
Reply with JSON only: {"score": 0|1|2, "reason": "<one sentence>"}"""

JUDGE_USER = """Claim: {claim}

Evidence:
{evidence}"""
