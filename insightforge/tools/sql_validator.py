"""Static + planner-level validation of model-generated SQL.

Order of checks (cheap to expensive):
  parse -> single statement -> read-only (no DML/DDL/admin nodes) -> no file/network table functions
  -> referenced tables exist -> row limit injected -> EXPLAIN (binder catches bad columns/types)
Every failure comes back as a message written for the repair prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from insightforge.db import Database, Schema

_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Merge", "Command", "Copy",
    "Attach", "Detach", "Pragma", "Install", "Load", "Set", "Use", "Transaction", "Commit", "Rollback",
    "TruncateTable", "Grant", "Revoke", "Analyze", "Export",
]
_FORBIDDEN_NODES = tuple(t for n in _FORBIDDEN_NODE_NAMES if isinstance(t := getattr(exp, n, None), type))

_DENY_FUNCS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_ndjson", "read_text",
    "read_blob", "parquet_scan", "csv_scan", "glob", "sniff_csv", "getenv", "load_extension", "readfile",
    "writefile", "fts_main", "sqlite_scan", "postgres_scan", "iceberg_scan", "delta_scan", "query_table",
}
_ALLOWED_TABLE_FUNCS = {"generate_series", "range", "unnest", "json_each", "json_tree"}


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    errors: list[str] = field(default_factory=list)
    stage: str = ""  # which check failed: parse | statement | readonly | function | table | explain
    tables: list[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors)


class SQLValidator:
    def __init__(self, db: Database, max_rows: int | None = 5000, explain: bool = True):
        self.db = db
        self.max_rows = max_rows
        self.use_explain = explain

    @property
    def schema(self) -> Schema:
        return self.db.schema()

    def validate(self, sql: str, enforce_limit: bool = True, max_rows: int | None = None) -> ValidationResult:
        cap = max_rows if max_rows is not None else self.max_rows
        sql = sql.strip().rstrip(";").strip()
        dialect = self.db.dialect
        try:
            stmts = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
        except sqlglot.errors.ParseError as e:
            return ValidationResult(False, sql, [f"SQL syntax error: {str(e).splitlines()[0]}"], "parse")
        if len(stmts) != 1:
            return ValidationResult(False, sql, [f"Expected exactly one statement, got {len(stmts)}."], "statement")
        root = stmts[0]

        if not isinstance(root, exp.Query):
            return ValidationResult(False, sql, [f"Only SELECT queries are allowed (got {type(root).__name__})."],
                                    "readonly")
        for node in root.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                return ValidationResult(False, sql, [f"Forbidden operation: {type(node).__name__}."], "readonly")

        for fn in root.find_all(exp.Func):
            name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
            if name in _DENY_FUNCS or name.startswith("read_"):
                return ValidationResult(False, sql, [f"Function {name}() is not allowed (file/network access)."],
                                        "function")

        cte_names = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
        referenced, unknown = [], []
        for t in root.find_all(exp.Table):
            if not isinstance(t.this, exp.Identifier):
                fname = (t.this.sql_name() if isinstance(t.this, exp.Func) else str(t.this)).lower()
                if fname not in _ALLOWED_TABLE_FUNCS:
                    return ValidationResult(False, sql, [f"Table function {fname} is not allowed."], "function")
                continue
            name = t.name
            if name.lower() in cte_names:
                continue
            real = self.schema.table(name)
            if real is None:
                unknown.append(name)
            elif real.name not in referenced:
                referenced.append(real.name)
        if unknown:
            avail = ", ".join(sorted(self.schema.tables))
            return ValidationResult(False, sql, [f"Unknown table(s): {', '.join(sorted(set(unknown)))}. "
                                                 f"Available tables: {avail}."], "table")

        out_sql = self._apply_limit(root, dialect, cap) if (enforce_limit and cap) else sql

        if self.use_explain:
            try:
                self.db.explain(out_sql)
            except Exception as e:
                return ValidationResult(False, out_sql, [f"Database rejected the query: {_short(e)}"], "explain",
                                        referenced)
        return ValidationResult(True, out_sql, [], "ok", referenced)

    @staticmethod
    def _apply_limit(root: exp.Query, dialect: str, cap: int) -> str:
        limit = root.args.get("limit")
        if limit is not None:
            try:
                if int(limit.expression.name) <= cap:
                    return root.sql(dialect=dialect)
            except (AttributeError, ValueError):
                pass
            return root.limit(cap).sql(dialect=dialect)
        if isinstance(root, exp.Select):
            return root.limit(cap).sql(dialect=dialect)
        # set operations: wrap so the cap applies to the whole union
        return f"SELECT * FROM ({root.sql(dialect=dialect)}) AS _q LIMIT {cap}"


def _short(e: Exception, n: int = 400) -> str:
    s = str(e).strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."
