"""Schema retrieval: pick the tables a question needs, instead of pasting the whole schema.

Three signals:
  1. BM25 over column "documents" (table + column name + description).
  2. A value index: question n-grams that exactly match stored cell values (e.g. 'boleto', 'SP').
     Matched values are also shown to the SQL model, which fixes a large class of WHERE-clause errors.
  3. Join-path completion: if two chosen tables only connect through a third, the third is added.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from insightforge.db import Database, Schema

_STOP = set("""a an the of in on for to and or by with from at as is are was were be been what which who whom
how many much does do did per each all any that this these those than then there their its it into
show list give find get me number count total average avg top most least highest lowest""".split())


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    toks = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for q in query:
                if q in tf:
                    f = tf[q]
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


@dataclass
class RetrievalResult:
    tables: list[str]
    scores: dict[str, float]
    matched_values: list[tuple[str, str, str]] = field(default_factory=list)  # (table, column, value)
    used_retrieval: bool = True

    def value_hints(self) -> str:
        if not self.matched_values:
            return ""
        return "\n".join(f"-- value {v!r} appears in {t}.{c}" for t, c, v in self.matched_values[:12])


class SchemaRetriever:
    def __init__(self, db: Database, max_values_per_column: int = 2000, max_value_len: int = 60):
        self.db = db
        self.schema: Schema = db.schema()
        self._cols: list[tuple[str, str]] = []
        docs = []
        for t in self.schema.tables.values():
            tdoc = tokenize(t.name) + tokenize(t.description)
            for c in t.columns:
                self._cols.append((t.name, c.name))
                docs.append(tdoc + tokenize(c.name) * 2 + tokenize(c.description))
        self.bm25 = BM25(docs)
        self.max_values = max_values_per_column
        self.max_value_len = max_value_len
        self._value_index: dict[str, list[tuple[str, str, str]]] | None = None
        self._lock = threading.Lock()

    # ---- value index -------------------------------------------------------------------------
    def _build_value_index(self) -> dict[str, list[tuple[str, str, str]]]:
        idx: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for t in self.schema.tables.values():
            for c in t.columns:
                ty = (c.type or "").upper()
                if (ty and not any(k in ty for k in ("CHAR", "TEXT", "STRING"))) or re.search(r"_id$", c.name.lower()):
                    continue
                try:
                    rows = self.db.execute(
                        f'SELECT DISTINCT "{c.name}" FROM "{t.name}" WHERE "{c.name}" IS NOT NULL '
                        f'AND LENGTH("{c.name}") <= {self.max_value_len} LIMIT {self.max_values + 1}',
                        timeout_s=5.0).rows
                except Exception:
                    continue
                if len(rows) > self.max_values:  # free text / high-cardinality: skip
                    continue
                for (v,) in rows:
                    if isinstance(v, str) and len(v.strip()) >= 2 and not re.fullmatch(r"[0-9a-fA-F-]{16,}", v):
                        idx[_norm(v)].append((t.name, c.name, v))
        return idx

    def match_values(self, question: str) -> list[tuple[str, str, str]]:
        with self._lock:
            if self._value_index is None:
                self._value_index = self._build_value_index()
        words = re.findall(r"[\w'.-]+", question)
        quoted = re.findall(r"['\"]([^'\"]{2,60})['\"]", question)
        grams = {_norm(q) for q in quoted}
        for n in (1, 2, 3, 4):
            for i in range(len(words) - n + 1):
                g = " ".join(words[i : i + n])
                if n == 1 and (g.lower() in _STOP or len(g) < 2):
                    continue
                grams.add(_norm(g))
        hits = []
        for g in grams:
            hits.extend(self._value_index.get(g, []))
        # longer matches first; they're more specific
        return sorted(set(hits), key=lambda h: -len(h[2]))

    # ---- retrieval ---------------------------------------------------------------------------
    def retrieve(self, question: str, hint: str = "", top_k: int = 4) -> RetrievalResult:
        all_tables = list(self.schema.tables)
        values = self.match_values(question + " " + hint)
        if len(all_tables) <= top_k:
            return RetrievalResult(all_tables, {t: 0.0 for t in all_tables}, values, used_retrieval=False)

        q = tokenize(question + " " + hint)
        col_scores = self.bm25.scores(q)
        per_table: dict[str, list[float]] = defaultdict(list)
        for (t, _), s in zip(self._cols, col_scores):
            per_table[t].append(s)
        qset = set(q)
        scores = {}
        for t, ss in per_table.items():
            ss = sorted(ss, reverse=True)
            name_hit = 2.0 * len(qset & set(tokenize(t)))
            scores[t] = ss[0] + 0.3 * sum(ss[1:3]) + name_hit
        for t, _, _ in values:
            scores[t] = scores.get(t, 0) + 3.0

        ranked = sorted(scores, key=scores.get, reverse=True)
        chosen = [t for t in ranked[:top_k] if scores[t] > 0] or ranked[:top_k]
        chosen = self._complete_join_paths(chosen)
        return RetrievalResult(chosen, {t: round(scores[t], 3) for t in chosen}, values)

    def _complete_join_paths(self, chosen: list[str]) -> list[str]:
        """Add intermediate tables so every chosen table is reachable from the top-ranked one."""
        out = list(chosen)
        adj = {t: self.schema.neighbours(t) for t in self.schema.tables}
        root = out[0]
        for target in chosen[1:]:
            path = _bfs(adj, root, target)
            if path:
                for t in path:
                    if t not in out:
                        out.append(t)
        return out

    def full(self, question: str = "", hint: str = "") -> RetrievalResult:
        """Ablation baseline: whole schema, no value hints."""
        tabs = list(self.schema.tables)
        return RetrievalResult(tabs, {t: 0.0 for t in tabs}, [], used_retrieval=False)


def _bfs(adj: dict[str, set[str]], src: str, dst: str) -> list[str] | None:
    from collections import deque

    prev = {src: None}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if u == dst:
            path = []
            while u is not None:
                path.append(u)
                u = prev[u]
            return path[::-1]
        for v in adj.get(u, ()):
            if v not in prev:
                prev[v] = u
                dq.append(v)
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())
