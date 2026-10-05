"""FastAPI service.   uvicorn insightforge.api.server:app --reload"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from insightforge import __version__, runtime

app = FastAPI(title="InsightForge", version=__version__,
              description="Evidence-grounded analytics agent: every claim links to the query that produced it.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class SQLRequest(BaseModel):
    sql: str = Field(min_length=6, max_length=20000)


@app.get("/health")
def health() -> dict:
    s = runtime.settings()
    return {"status": "ok", "version": __version__, "db": s.db_path, "model": s.provider_label}


@app.get("/schema")
def schema() -> dict:
    sc = runtime.database().schema()
    return {"dialect": sc.dialect,
            "tables": {t.name: {"rows": t.row_count, "description": t.description,
                                "columns": [{"name": c.name, "type": c.type, "description": c.description}
                                            for c in t.columns]} for t in sc.tables.values()}}


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    rep = await run_in_threadpool(runtime.agent().ask, req.question)
    return rep.to_dict() | {"markdown": rep.to_markdown()}


@app.post("/sql")
async def sql(req: SQLRequest) -> dict:
    """Run caller-supplied SQL through the same read-only validator the agent uses."""
    run = await run_in_threadpool(runtime.agent().sql.run_sql_direct, req.sql)
    if not run.ok:
        raise HTTPException(400, detail=run.error)
    df = run.result.df
    return {"sql": run.sql, "row_count": len(df), "columns": list(df.columns),
            "rows": json.loads(df.head(1000).to_json(orient="records", date_format="iso", default_handler=str))}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if not run_id.replace("-", "").isalnum():
        raise HTTPException(400, "bad run id")
    p = runtime.settings().runs_dir / run_id / "report.json"
    if not p.exists():
        raise HTTPException(404, "run not found")
    return json.loads(p.read_text(encoding="utf-8"))
