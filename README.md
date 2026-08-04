# RTL Debug Agent

An AI agent that debugs RTL simulation failures by querying a **deterministic graph
representation** of the design ([`RTLGraph`](https://github.com/bhavanibedreshankar/RTLGraph)) —
signal drivers, fanin/fanout, clock/reset domains — instead of relying on embeddings/RAG.
Given a failing test's log directory, it traces the failure through the graph, cross-references
the design spec, proposes a minimal RTL patch with a quantified confidence score, and — after a
human approval checkpoint — applies it on a new git branch and reruns the test to verify the fix.

**Proven end-to-end**: pointed at a real 16×16 systolic-array accelerator design
([`ai-accelerator-development-platform`](https://github.com/bhavanibedreshankar/tpe-tensor-processing-engine))
with intentionally injected bugs, the agent independently localized a write-1-to-clear
register bug to the exact source line, proposed a one-line fix, and verified it turned a
failing test green — without ever reading the project's answer key.

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
docs/              generated dashboard, served via GitHub Pages
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
python -m dashboard.generate --runs-root data/runs --graph-eval data/graph_eval_report.json --out docs/index.html
```

## Tech stack

Python 3.11+, LangGraph, LangChain, LangMem, Anthropic Claude, Streamlit, SQLite
(LangGraph checkpoints + LangMem procedural memory), Pydantic, httpx.
