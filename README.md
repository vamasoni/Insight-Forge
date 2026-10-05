# InsightForge

**An analytics agent that answers business questions with SQL and statistics, where every claim in the report links to the query or test that produced it — plus an evaluation harness that blocks changes that make it worse.**

"Chat with your data" tools fail in three quiet ways: wrong SQL, the wrong statistical test, and conclusions the data doesn't support. InsightForge attacks each one: validated SQL with self-repair, vetted statistical tests run in a sandbox, and a critic that rejects any claim whose numbers, citations or causal language the evidence can't back. A benchmark suite runs in CI so a prompt or model change that lowers accuracy fails the build.

```
Question ──► FastAPI / CLI / MCP / Streamlit
               │
               ▼
        LangGraph agent
        ├─ Planner ........ splits the question, picks tests; later writes the findings
        ├─ SQL tool ....... schema retrieval → generate → validate → execute → repair
        ├─ Analysis tool .. sandboxed pandas/SciPy/statsmodels, Plotly charts
        └─ Critic ......... deterministic checks + LLM check; one revision loop
               │
               ▼
   Report: findings [F1..Fn] → evidence [Q = query, C = calculation, V = chart]
               │
   Langfuse traces (or runs/traces.jsonl)   ·   Eval harness + GitHub Action gate
```

## Results

| Metric | Value | How measured |
|---|---|---|
| BIRD dev execution accuracy, retrieval + repair | **TODO** | `evals.bird.run_bird`, full dev set (1,534 Q) |
| BIRD EX, full schema, no repair (baseline) | **TODO** | same, `--configs full` |
| Repair success rate (Q needing repair that end correct) | **TODO** | from the BIRD summary |
| Olist: answer present in evidence / stated correctly in report | **TODO** / **TODO** | `evals.olist.run_olist --mode agent` |
| Critic recall on corrupted claims (deterministic layer) | 99.5% (216/217), 0% false positives on 43 clean claims | `evals.critic_eval`, synthetic Olist |
| Judge–human agreement (quadratic-weighted Cohen's κ) | **TODO** | `evals.judge.judge calibrate`, ≥100 hand-labelled claims |
| Cost per question | **TODO** | tracked per LLM call |

Fill the TODOs by running the commands below; don't estimate them. The critic number is real but optimistic. It covers the deterministic layer only, on synthetic data, against corruption types I designed. The one miss is a correlation of about 0.001 written as "0.0%", which no number check can distinguish.

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
copy .env.example .env        # then set LLM_API_KEY (and LLM_BASE_URL for Mistral/vLLM)

# Data option A: real Olist (needs a Kaggle API token in %USERPROFILE%\.kaggle\kaggle.json)
pip install kaggle  
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist_csv --unzip
python scripts/load_olist.py --csv-dir data/olist_csv --out data/olist.duckdb

# Data option B: synthetic Olist-shaped data (no account needed; what CI uses)
python scripts/make_synthetic_olist.py --out data/olist_csv_synth
python scripts/load_olist.py --csv-dir data/olist_csv_synth --out data/olist.duckdb

pytest -q                                            # 46 tests, no API key needed
python -m insightforge ask "Do late deliveries get worse review scores?"
uvicorn insightforge.api.server:app --reload         # http://localhost:8000/docs
streamlit run app/streamlit_app.py
python -m insightforge.mcp_server                    # MCP over stdio
```

On macOS/Linux, use `source .venv/bin/activate` and `cp`.

### MCP client config (e.g. Claude Desktop)
```json
{"mcpServers": {"insightforge": {
  "command": "C:\\path\\to\\insightforge\\.venv\\Scripts\\python.exe",
  "args": ["-m", "insightforge.mcp_server"],
  "env": {"DB_PATH": "C:\\path\\to\\insightforge\\data\\olist.duckdb"}}}}
```
Tools: `list_tables`, `describe_table`, `find_relevant_tables`, `run_sql`, `run_analysis`, `ask`. They go through the same validator and sandbox as the agent.

## Evaluation

```powershell
python scripts/download_bird.py                                         # BIRD dev (~1,534 questions, SQLite)
python -m evals.bird.run_bird --configs full full+repair retrieval retrieval+repair --workers 6
python -m evals.olist.run_olist --mode sql                              # Olist EX (extra columns allowed)
python -m evals.olist.run_olist --mode agent                            # full pipeline + contract checks
python -m evals.critic_eval --db data/olist.duckdb [--llm]
python -m evals.judge.judge export ; <label human_score in results/labels.csv>
python -m evals.judge.judge score  ; python -m evals.judge.judge calibrate
```

**Deterministic checks.** These cover:
- Execution accuracy, using BIRD's official set-equality rule.
- Schema-valid outputs: planner and synthesis JSON are validated with pydantic, and failures are counted.
- Tool-call correctness: every sub-question produced executed SQL, every requested test ran, and every claim cites evidence that exists.
- Retrieval table recall: did retrieval include every table the gold SQL uses?

**LLM-as-judge.** It uses a 0/1/2 faithfulness rubric. It is only trusted as a gate if quadratic-weighted κ against your own labels is at least 0.6; the calibrate command prints the verdict.

**Regression gate (`evals/regression.py`).** A 150-question BIRD subset has a 95% CI of roughly ±7 points, so "accuracy went down" alone is noise. The gate fails on either condition:
1. A drop larger than a hard floor.
2. McNemar's exact test on paired per-question outcomes shows significantly more regressions than fixes. This catches consistent small regressions that the floor misses.

The subset is a fixed stratified sample (same seed, same questions), which is what makes the paired test valid.

**CI** (`.github/workflows/ci.yml`) has two jobs:
- **Test job:** unit tests, then the critic eval gated against `evals/baselines/critic.json`. It needs no secrets.
- **LLM job:** runs only when the `LLM_API_KEY` secret exists. It runs the BIRD subset and a synthetic-Olist agent eval, gates both, and writes a summary table on the run page.

The first time the LLM job runs there is no baseline, so the gate warns and passes. Commit one with:
```powershell
python -m evals.regression --current results/bird_retrieval_repair_summary.json --baseline evals/baselines/bird_retrieval_repair.json --update-baseline
```

## Design decisions

- **Numbers are checked by code, not by an LLM.** The critic extracts every number in a claim and requires the cited evidence to contain it at the claim's own rounding precision. It also handles % vs fraction. An early version used a 0.5% relative tolerance, and the critic eval caught it letting wrong numbers through by coincidentally landing near other values in the same table. Tightening it took recall from 98.6% to 99.5% with no new false positives.
- **Causal language is rejected outright.** Olist and BIRD are observational, so "drives", "causes" and "leads to" fail the check. "Statistically significant" requires a cited test with p < 0.05.
- **Vetted tests instead of free-form code by default.** The planner picks from Welch t-test, chi-square, correlation, OLS with HC3 errors, and robust-z anomalies. Each returns the effect size and CI, not just p. A free-form pandas mode exists, behind an AST allowlist.
- **Analysis inputs aren't capped.** Display queries cap at 5,000 rows, but queries feeding a statistical test use a separate 500k cap. A test on a truncated sample is silently wrong; this bug was found during testing.
- **Read-only is enforced twice.** It's enforced at the connection level (DuckDB `read_only` with external access disabled; SQLite `mode=ro`) and again in the sqlglot validator.
- **Value index for retrieval.** Question phrases that match stored cell values ("boleto", "SP") are passed to the model as exact literals. This targets the WHERE-clause errors that dominate text-to-SQL failures.

## Limitations

- **Retrieval is lexical** (BM25 + value matching). It can't know that "revenue" means `SUM(order_items.price)` unless that's in a table or column comment. With `top_tables=2`, such questions miss a table. Adding embedding-based retrieval is the natural next step.
- **BIRD has few tables per database.** Retrieval may not help EX there and could even hurt; the ablation will show which. Report it either way.
- **The sandbox is not hostile-grade.** It's a subprocess with a timeout, rlimits (POSIX only; on Windows only the timeout applies), an empty environment and an AST allowlist. For untrusted multi-tenant use, run it in a container with no network.
- **The 50 Olist gold queries are my interpretation.** They are executed against your data at eval time, so check them on the real dataset before calling them verified answers. Real Olist has multi-payment orders and duplicate review IDs; decide how each question should treat them.

## Optional next step: small open model for SQL

The provider layer already talks to any OpenAI-compatible endpoint. The plan: LoRA-tune a small coder model on BIRD train, quantize it, serve it with `vllm serve`, and set `LLM_BASE_URL` to it. Then rerun the BIRD ablation to compare EX, cost per query and p50/p95 latency against the API model. Latency and token counts are already recorded per question.

## Repo layout

```
insightforge/   config, llm/ (providers), db/, retrieval/, tools/ (sql_validator, sql_tool, analysis_tool,
                stats_lib, sandbox_runner), evidence/, agent/ (graph, critic, report), api/, mcp_server.py
evals/          metrics.py, bird/, olist/ (50 questions), critic_eval.py, judge/, regression.py, baselines/
scripts/        load_olist.py, make_synthetic_olist.py, download_bird.py
app/            streamlit_app.py
tests/          46 tests, fake LLM, no network
```
