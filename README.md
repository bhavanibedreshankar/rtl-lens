# RTL-Lens

**An AI RTL debug agent that uses graph RAG — not embeddings — as its primary tool for
tracing simulation failures to their root cause.**

### 📊 [View live debug reports →](https://rtl-lens-bhavani89.vercel.app/)

RTL-Lens debugs RTL simulation failures by querying a **deterministic graph
representation** of the design ([`RTLGraph`](https://github.com/bhavanibedreshankar/RTLGraph)) —
signal drivers, fanin/fanout, clock/reset domains — instead of vector-embedding-based
retrieval. Given a failing test's log directory, it traces the failure through the graph, cross-references
the design spec, proposes a minimal RTL patch with a quantified confidence score, and — after a
human approval checkpoint — applies it on a new git branch and reruns the test to verify the fix.

## Evaluation: what it fixed, what it didn't

Run blind (never reading the demo repo's answer key) against a real 16×16 systolic-array
accelerator design ([`tpe-tensor-processing-engine`](https://github.com/bhavanibedreshankar/tpe-tensor-processing-engine))
with 7 intentionally injected RTL bugs, graded *after the fact* by an offline script that
checks the agent's independent conclusion against the answer key. This is the honest
scorecard, not a cherry-picked one — see the [live report](https://rtl-lens-bhavani89.vercel.app/)
for full investigation traces on every one of these:

| Test | Result | Notes |
|---|---|---|
| `top.irq_independent_clear_test` | ✅ Fixed & verified | W1C register mask bug, exact line match |
| `matrix_engine.matmul_overflow_test` | ✅ Fixed & verified | Asymmetric saturation clamp, exact line match |
| `top.matmul_full_width_test` | ✅ Fixed & verified | Off-by-one boundary check, exact line match |
| `pmu.latency_test` | ✅ Fixed & verified | NBA-ordering latency bug — RTL source had a comment giving away the answer, so this one doesn't demonstrate blind debugging as cleanly as the others |
| `dma.dma_multiburst_write_test` | ⚠️ Not resolved | Landed on the right file across several attempts but never the right line; rejected rather than shipped |
| `matrix_engine.matmul_random_test` (bugs targeting `dim_k`/seed-chain off-by-ones) | ⚠️ Not resolved | Repeatedly diagnosed a *different*, real bug in the same block instead |

**4 of 6 targeted bugs fixed and verified**, 2 honestly rejected rather than shipped
incorrectly — including one case where a wrong patch would have scored "correct" by a naive
line-proximity check but was actually incomplete and wouldn't compile (an undefined state
value), caught only because the human-approval checkpoint exists.

## How it works

```
failing test logs → ingest → investigate (graph + spec + RTL tools, looped)
                  → synthesize hypothesis → score confidence
                  → [low confidence? back to investigate]
                  → human checkpoint (approve / reject / more evidence)
                  → apply patch on a new branch → rerun test → verified
```

Built on [LangGraph](https://langchain-ai.github.io/langgraph/) for the state machine and
human-in-the-loop checkpoints (a genuine `interrupt()`, resumable from a separate process —
`debug` can run unattended, then `approve`/`reject` days later), [LangChain](https://python.langchain.com/)
for tool/LLM abstractions, and [LangMem](https://langchain-ai.github.io/langmem/) for
token-conserving memory (short-term summarization of long investigations, plus a persistent
store of which tool sequences found root causes for past failure categories).

## Features

- **Blind diagnosis** — the diagnosis loop's tools are scoped so the project's bug-list answer
  key is structurally unreachable (see `agent/tools/rtl_tools.py` / `spec_tools.py`); a separate
  offline `grader.py` checks the agent's independent conclusion against it *after the fact*.
- **Quantified confidence** — combines the model's self-reported confidence with deterministic
  grounding checks (did it actually read the file it's patching? did it use the graph? the spec?)
  into one score, with low-confidence hypotheses automatically bounced back for more evidence
  before ever reaching a human.
- **Token tracking & conservation** — every LLM call is attributed to its graph node and
  accumulated across a run's multiple CLI invocations, with a configurable per-run budget;
  cheaper/faster models handle routine tool selection, a stronger model only for the final
  hypothesis; repeated graph queries are cached; long tool-call histories get LangMem-summarized.
- **Human-in-the-loop checkpoints** — autonomous up to a patch proposal, then a real pause
  (`interrupt()`) for approve / reject / send-back-for-more-evidence, from the CLI or the GUI.
- **Config-driven, portable to other designs** — nothing under `agent/` hardcodes a path, URL,
  or RTL repo; `config/<project>.yaml` is the only thing that changes (see `config/tpe.yaml`).
- **Graph DB evaluation harness** (`graph_eval/`) — scores any graph database's structural
  health, retrieval accuracy against known-answer queries, and RTL source coverage, independent
  of any specific debug run.
- **GUI chat mode** (`chat_ui/`) — a Streamlit app that streams the investigation live and
  surfaces the same approve/reject/more-evidence checkpoint as inline buttons.
- **Reviewer dashboard** (`docs/index.html`) — generated from every run's trail: suite, failing
  test, fail signature, the agent's investigation plan and step-by-step retrieved evidence,
  hypothesis, confidence, the proposed patch, the answer-key check, and token cost.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set ANTHROPIC_API_KEY

python -m agent.cli debug \
  --failure-dir /path/to/SIM_LOGS/WORK/smoke/<block>.<test>/rtl_sim \
  --config config/tpe.yaml
# review data/runs/<run_id>/patch.diff, then:
python -m agent.cli approve --run-id <run_id>
```

Or drive the same flow interactively:

```bash
streamlit run chat_ui/app.py
```

## Project structure

```
agent/            LangGraph state machine, nodes, tools, memory, CLI
  nodes/            ingest -> investigate -> synthesize -> score_confidence -> checkpoint -> apply_and_verify
  tools/             graph_tools, spec_tools, rtl_tools, sim_tools, git_tools, log_tools
config/            per-project adapter configs (schema.py + tpe.yaml)
grader.py          offline answer-key check, never imported by agent/*
graph_eval/        graph database health/accuracy scoring harness
chat_ui/           Streamlit GUI
dashboard/         static reviewer dashboard generator
data/runs/         per-run evidence trail, patch, token report, grade (git-tracked)
docs/              generated dashboard, deployed to Vercel (see link above)
```

## Pointing at a different RTL design

Copy `config/tpe.yaml` to `config/<yours>.yaml` and point it at your RTL repo, your graph DB's
REST endpoint (must implement the contract in `agent/tools/graph_tools.py`), and your
simulation run command. See that file for the full schema.

## Evaluating a graph database

```bash
python -m graph_eval.run --config config/tpe.yaml --report docs/graph_eval_report.html
```

## Regenerating the dashboard

```bash
python -m dashboard.generate \
  --runs-root data/runs --graph-eval data/graph_eval_report.json \
  --regression-xml /path/to/regression.xml --suite smoke \
  --actions-url https://github.com/<owner>/<repo>/actions --out-dir docs
```

Writes `docs/index.html` (the regression-level index — picked suite, GitHub Actions link,
full pass/fail table) plus one `docs/reports/<test>.html` per debugged test. Redeploy with
`vercel deploy docs --prod` (see `docs/index.html`'s live link above).

## Tech stack

Python 3.11+, LangGraph, LangChain, LangMem, Anthropic Claude, Streamlit, SQLite
(LangGraph checkpoints + LangMem procedural memory), Pydantic, httpx.
