"""MCP server exposing the database and analysis tools to any MCP client (Claude Desktop, Cursor, ...).

    python -m insightforge.mcp_server                       # stdio (for desktop clients)
    python -m insightforge.mcp_server --http --port 8765    # streamable HTTP

Every tool goes through the same validator/sandbox as the agent, so an MCP client gets no extra power.
"""
from __future__ import annotations

import argparse
import json

try:  # mcp >= 2
    from mcp.server import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from insightforge import runtime
from insightforge.tools.analysis_tool import AnalysisTool
from insightforge.tools.stats_lib import REGISTRY

mcp = _Server("insightforge", instructions=(
    "Read-only analytics over a relational database. Call list_tables/describe_table first, then run_sql. "
    "Use run_analysis for statistical tests on a SQL result. Use ask for a full evidence-linked report."))

_MAX_RETURN_ROWS = 200


@mcp.tool()
def list_tables() -> str:
    """List tables with row counts and descriptions."""
    sc = runtime.database().schema()
    return json.dumps([{"table": t.name, "rows": t.row_count, "description": t.description}
                       for t in sc.tables.values()], indent=1)


@mcp.tool()
def describe_table(table: str) -> str:
    """Show a table's columns, types, descriptions, sample values and join keys."""
    sc = runtime.database().schema()
    t = sc.table(table)
    if t is None:
        return f"Unknown table {table!r}. Tables: {', '.join(sc.tables)}"
    return sc.render([t.name, *sorted(sc.neighbours(t.name))])


@mcp.tool()
def find_relevant_tables(question: str) -> str:
    """Schema retrieval: which tables a natural-language question needs, plus matched literal values."""
    r = runtime.agent().sql.retriever.retrieve(question)
    return json.dumps({"tables": r.tables, "scores": r.scores,
                       "matched_values": [{"table": t, "column": c, "value": v} for t, c, v in r.matched_values[:15]]})


@mcp.tool()
def run_sql(sql: str) -> str:
    """Execute ONE read-only SELECT. Validated (no writes, no file access, row-limited) before running."""
    run = runtime.agent().sql.run_sql_direct(sql)
    if not run.ok:
        return f"ERROR ({run.attempts[-1].stage}): {run.error}"
    df = run.result.df
    return json.dumps({"sql_executed": run.sql, "row_count": len(df), "columns": list(df.columns),
                       "rows": json.loads(df.head(_MAX_RETURN_ROWS).to_json(orient="records", date_format="iso",
                                                                            default_handler=str)),
                       "truncated_in_response": len(df) > _MAX_RETURN_ROWS}, default=str)


@mcp.tool()
def run_analysis(sql: str, method: str, args: dict) -> str:
    """Run a statistical test on the result of a SELECT.
    method: t_test | chi_square | correlation | regression | anomalies | describe
    args examples: t_test {"value": "review_score", "group": "delivery_status"};
    correlation {"x": "a", "y": "b", "method": "spearman"}; regression {"y": "a", "x": ["b", "c"]}"""
    if method not in REGISTRY:
        return f"Unknown method. Choose from {sorted(REGISTRY)}"
    s = runtime.settings()
    run = runtime.agent().sql.run_sql_direct(sql, max_rows=s.analysis_max_rows)  # tests need all rows
    if not run.ok:
        return f"SQL ERROR: {run.error}"
    res = AnalysisTool(timeout_s=s.analysis_timeout_s).run_helper(method, args, {"Q1": run.result.df})
    return json.dumps(res.outputs if res.ok else {"error": res.error}, default=str)


@mcp.tool()
def ask(question: str) -> str:
    """Full agent: plan, query, test, write a report where every claim cites its evidence, critic-checked."""
    return runtime.agent().ask(question).to_markdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.http:
        mcp.run(transport="streamable-http", port=a.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
