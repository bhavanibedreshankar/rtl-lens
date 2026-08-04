from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agent.logging_config import get_logger, log_event
from agent.state import DebugState
from agent.tools.log_tools import parse_failure
from config.schema import ProjectConfig

logger = get_logger("nodes.ingest_failure")

_SYSTEM_PROMPT = """You are an RTL debug agent. You investigate a failing hardware \
simulation test by querying a deterministic design graph (signal drivers, fanin/fanout, \
clock/reset domains — NOT an embedding search) and the design spec, then propose a minimal \
RTL patch.

Rules:
- You do NOT have access to any "known bugs" list. Diagnose from first principles: the \
failure signature, the graph, the spec, and RTL source you read yourself.
- Prefer graph_trace_driver / graph_fanin to find what logic produces a signal before \
reading source blindly.
- Use spec_search to confirm what the RTL is SUPPOSED to do before concluding it's wrong.
- Use rtl_read_file to see the actual source around a suspect driver before proposing a fix.
- Once you have enough evidence to state a root cause with a concrete file:line and a \
minimal patch, say so plainly and stop calling tools — you will then be asked to produce \
the structured hypothesis.
- Keep tool arguments precise (exact signal/module names) to avoid wasted, unfocused calls — \
tokens are budgeted for this run."""


def build_failure_prompt(cfg: ProjectConfig, failure) -> str:
    return f"""A test failed in simulation. Investigate the root cause.

Test: {failure.test}
Block: {failure.block}
Error class: {failure.error_class}
Failure message: {failure.message}

Candidate signal/register names mentioned in the failure: {', '.join(failure.signals) or '(none extracted — read the excerpt below)'}

Relevant log excerpt:
```
{failure.raw_excerpt}
```

The RTL design root is: {cfg.rtl_repo.rtl_dir}/ (design name: {cfg.project}).
Investigate using your tools, then report your root-cause conclusion in plain text."""


def ingest_failure(state: DebugState, *, config: ProjectConfig) -> dict:
    failure_dir = Path(state["failure_dir"])
    failure = parse_failure(failure_dir, sim_cfg=config.sim)

    log_event(logger, "ingest_failure", test=failure.test, run_id=state["run_id"])

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=build_failure_prompt(config, failure)),
    ]

    return {
        "failure": failure,
        "messages": messages,
        "evidence": [],
        "investigation_steps": 0,
        "confidence_retries": 0,
        "phase": "diagnosing",
    }
