"""Read-only database access for DuckDB (Olist) and SQLite (BIRD), with schema introspection.

Read-only is enforced at the connection level here AND at the SQL level in tools/sql_validator.py.
"""
from __future__ import annotations

import csv
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


class QueryTimeout(Exception):
    pass


@dataclass
class Column:
    name: str
    type: str
    description: str = ""
    samples: list[Any] = field(default_factory=list)
    is_pk: bool = False


@dataclass
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str
    inferred: bool = False


@dataclass
class Table:
    name: str
    columns: list[Column]
    row_count: int = 0
    description: str = ""

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name.lower() == name.lower()), None)


@dataclass
class Schema:
    tables: dict[str, Table]
    foreign_keys: list[ForeignKey]
    dialect: str

    def table(self, name: str) -> Table | None:
        return next((t for n, t in self.tables.items() if n.lower() == name.lower()), None)

    def neighbours(self, table: str) -> set[str]:
        out = set()
        for fk in self.foreign_keys:
            if fk.table.lower() == table.lower():
                out.add(fk.ref_table)
            elif fk.ref_table.lower() == table.lower():
                out.add(fk.table)
        return out

    def render(self, tables: list[str] | None = None, max_samples: int = 3) -> str:
        """Compact DDL-style rendering used in prompts."""
        names = tables if tables is not None else list(self.tables)
        chosen = {n.lower() for n in names}
        parts = []
        for name in names:
            t = self.table(name)
            if t is None:
                continue
            lines = [f"CREATE TABLE {_q(t.name)} (  -- {t.row_count:,} rows" + (f"; {t.description}" if t.description else "")]
            for c in t.columns:
                note = []
                if c.is_pk:
                    note.append("PK")
                if c.description:
                    note.append(c.description)
                if c.samples:
                    note.append("e.g. " + ", ".join(_fmt(s) for s in c.samples[:max_samples]))
                lines.append(f"  {_q(c.name)} {c.type}," + (f"  -- {'; '.join(note)}" if note else ""))
            parts.append("\n".join(lines) + "\n);")
        fks = [fk for fk in self.foreign_keys if fk.table.lower() in chosen and fk.ref_table.lower() in chosen]
        if fks:
            parts.append("-- Join keys:\n" + "\n".join(
                f"-- {_q(f.table)}.{_q(f.column)} = {_q(f.ref_table)}.{_q(f.ref_column)}" for f in fks))
        return "\n\n".join(parts)


def _q(ident: str) -> str:
    return ident if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident) else f'"{ident}"'


def _fmt(v: Any) -> str:
    s = str(v)
    return repr(s[:40]) if isinstance(v, str) else s


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    elapsed_s: float

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=_dedupe(self.columns))

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _dedupe(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


class Database(ABC):
    dialect: str
    path: str

    def __init__(self, path: str, sample_values: int = 5, descriptions_dir: str | None = None):
        self.path = str(path)
        self.sample_values = sample_values
        self.descriptions_dir = descriptions_dir
        self._schema: Schema | None = None
        self._lock = threading.Lock()

    @abstractmethod
    def execute(self, sql: str, timeout_s: float = 30.0) -> QueryResult: ...

    def explain(self, sql: str, timeout_s: float = 10.0) -> None:
        """Plans the query without running it; raises on unknown tables/columns/type errors."""
        self.execute(f"EXPLAIN {sql}", timeout_s=timeout_s)

    @abstractmethod
    def _introspect(self) -> Schema: ...

    def schema(self) -> Schema:
        with self._lock:
            if self._schema is None:
                self._schema = self._introspect()
                _load_join_sidecar(self._schema, Path(self.path + ".joins.json"))
                _infer_foreign_keys(self._schema)
                if self.descriptions_dir:
                    load_bird_descriptions(self._schema, Path(self.descriptions_dir))
            return self._schema

    def _samples(self, table: str, col: str, ctype: str) -> list[Any]:
        if self.sample_values <= 0:
            return []
        t = (ctype or "").upper()
        if "BLOB" in t or (t and not any(k in t for k in ("CHAR", "TEXT", "STRING", "CLOB"))):
            return []  # only text columns: their values are what questions mention by name
        if re.search(r"(_id|_key|uuid|hash)$", col.lower()):
            return []
        try:
            res = self.execute(
                f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {self.sample_values}',
                timeout_s=5.0)
            vals = [r[0] for r in res.rows]
        except Exception:
            return []
        if vals and all(isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F-]{16,}", v) for v in vals):
            return []  # opaque identifiers only waste prompt tokens
        return vals


class DuckDBDatabase(Database):
    dialect = "duckdb"

    def __init__(self, path: str, **kw):
        super().__init__(path, **kw)
        import duckdb

        self._con = duckdb.connect(self.path, read_only=True)
        # no file/network access from SQL even if something slips past the validator
        for stmt in ("SET enable_external_access = false", "SET lock_configuration = true"):
            try:
                self._con.execute(stmt)
            except Exception:
                pass

    def execute(self, sql: str, timeout_s: float = 30.0) -> QueryResult:
        cur = self._con.cursor()
        timer = threading.Timer(timeout_s, cur.interrupt)
        t0 = time.perf_counter()
        timer.start()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        except Exception as e:
            if "interrupt" in str(e).lower():
                raise QueryTimeout(f"query exceeded {timeout_s}s") from e
            raise
        finally:
            timer.cancel()
            cur.close()
        return QueryResult(cols, [tuple(r) for r in rows], time.perf_counter() - t0)

    def _introspect(self) -> Schema:
        cols = self.execute(
            "SELECT table_name, column_name, data_type, COALESCE(comment, '') FROM duckdb_columns() "
            "WHERE schema_name='main' AND NOT internal ORDER BY table_name, column_index").rows
        tcomments = dict(self.execute(
            "SELECT table_name, COALESCE(comment, '') FROM duckdb_tables() WHERE schema_name='main'").rows)
        tables: dict[str, Table] = {}
        for tname, cname, ctype, comment in cols:
            t = tables.setdefault(tname, Table(tname, [], description=tcomments.get(tname, "")))
            t.columns.append(Column(cname, ctype, description=comment, samples=self._samples(tname, cname, ctype)))
        for t in tables.values():
            t.row_count = self.execute(f'SELECT COUNT(*) FROM "{t.name}"').rows[0][0]
        fks = []
        try:
            for tname, ccols, rtable, rcols in self.execute(
                    "SELECT table_name, constraint_column_names, referenced_table, referenced_column_names "
                    "FROM duckdb_constraints() WHERE constraint_type='FOREIGN KEY'").rows:
                fks += [ForeignKey(tname, c, rtable, r) for c, r in zip(ccols, rcols)]
        except Exception:
            pass
        return Schema(tables, fks, self.dialect)


class SQLiteDatabase(Database):
    dialect = "sqlite"

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{Path(self.path).as_posix()}?mode=ro", uri=True, check_same_thread=False)
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        return con

    def execute(self, sql: str, timeout_s: float = 30.0) -> QueryResult:
        con = self._connect()
        deadline = time.perf_counter() + timeout_s
        con.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
        t0 = time.perf_counter()
        try:
            cur = con.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise QueryTimeout(f"query exceeded {timeout_s}s") from e
            raise
        finally:
            con.close()
        return QueryResult(cols, rows, time.perf_counter() - t0)

    def _introspect(self) -> Schema:
        names = [r[0] for r in self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").rows]
        tables, fks = {}, []
        for n in names:
            info = self.execute(f'PRAGMA table_info("{n}")').rows  # cid, name, type, notnull, dflt, pk
            cols = [Column(r[1], r[2] or "TEXT", is_pk=bool(r[5]), samples=self._samples(n, r[1], r[2] or "TEXT"))
                    for r in info]
            try:
                cnt = self.execute(f'SELECT COUNT(*) FROM "{n}"', timeout_s=10).rows[0][0]
            except Exception:
                cnt = 0
            tables[n] = Table(n, cols, row_count=cnt)
            for r in self.execute(f'PRAGMA foreign_key_list("{n}")').rows:  # id, seq, table, from, to, ...
                ref_col = r[4]
                if ref_col is None:  # FK to the referenced table's PK
                    pk = [c[1] for c in self.execute(f'PRAGMA table_info("{r[2]}")').rows if c[5]]
                    ref_col = pk[0] if pk else r[3]
                fks.append(ForeignKey(n, r[3], r[2], ref_col))
        return Schema(tables, fks, self.dialect)


def _load_join_sidecar(schema: Schema, path: Path) -> None:
    """Optional <db>.joins.json listing joins that naming conventions can't reveal."""
    if path.exists():
        import json

        for j in json.loads(path.read_text()):
            if schema.table(j["table"]) and schema.table(j["ref_table"]):
                schema.foreign_keys.append(ForeignKey(j["table"], j["column"], j["ref_table"], j["ref_column"]))


def _infer_foreign_keys(schema: Schema) -> None:
    """Add join hints for identically named *_id / *_key columns when no FK is declared (e.g. Olist CSVs)."""
    declared = {(f.table.lower(), f.column.lower()) for f in schema.foreign_keys}
    by_col: dict[str, list[str]] = {}
    for t in schema.tables.values():
        for c in t.columns:
            if re.search(r"(_id|_key|_prefix)$", c.name.lower()):
                by_col.setdefault(c.name.lower(), []).append(t.name)
    for col, tabs in by_col.items():
        if len(tabs) < 2:
            continue
        prefix = re.sub(r"(_id|_key|_prefix)$", "", col)
        owners = [t for t in tabs if t.lower() in (prefix, prefix + "s", prefix + "es")]
        pk_owners = [t for t in tabs if (c := schema.tables[t].column(col)) is not None and c.is_pk]
        anchor = (owners or pk_owners or tabs)[0]
        for other in [t for t in tabs if t != anchor]:
            if (other.lower(), col) not in declared and (anchor.lower(), col) not in declared:
                schema.foreign_keys.append(ForeignKey(other, col, anchor, col, inferred=True))


def load_bird_descriptions(schema: Schema, desc_dir: Path) -> None:
    """BIRD ships database_description/<table>.csv with column descriptions and value hints."""
    if not desc_dir.exists():
        return
    for f in desc_dir.glob("*.csv"):
        t = schema.table(f.stem)
        if t is None:
            continue
        for enc in ("utf-8-sig", "latin-1"):
            try:
                with f.open(encoding=enc) as fh:
                    rows = list(csv.DictReader(fh))
                break
            except UnicodeDecodeError:
                continue
        else:
            continue
        for row in rows:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            c = t.column(row.get("original_column_name", ""))
            if c is None:
                continue
            bits = [row.get("column_description", ""), row.get("value_description", "")]
            full = row.get("column_name", "")
            if full and full.lower() != c.name.lower():
                bits.insert(0, full)
            c.description = " | ".join(b.replace("\n", " ")[:200] for b in bits if b)


def open_database(path: str, dialect: str | None = None, **kw) -> Database:
    d = dialect or ("sqlite" if Path(path).suffix in (".sqlite", ".db", ".sqlite3") else "duckdb")
    return SQLiteDatabase(path, **kw) if d == "sqlite" else DuckDBDatabase(path, **kw)
