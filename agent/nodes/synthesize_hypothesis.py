from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agent.logging_config import get_logger, log_event
from agent.state import DebugState, Hypothesis

logger = get_logger("nodes.synthesize_hypothesis")

_INSTRUCTION = """Based on your investigation above, produce your final diagnosis.

CRITICAL: base the patch ONLY on source text you actually saw via `rtl_read_file` in this \
conversation. If you have not read the exact lines you're about to change, call \
`rtl_read_file` again before answering — do not reconstruct or guess the surrounding code \
from the module's general shape or from what a "typical" implementation might look like. \
Every context line in your diff must be a verbatim copy of a line you read, or `git apply` \
will simply reject the patch.

Requirements for `patch_diff`:
- A minimal unified diff (as `git apply` would accept), touching only the RTL file(s) \
you identified as the root cause.
- Use paths relative to the repo root exactly as you read them (e.g. \
`rtl/command_processor/tpe_cmd_proc.sv`), with standard `--- a/<path>` / `+++ b/<path>` \
headers and `@@ -l,c +l,c @@` hunk headers whose line numbers match what `rtl_read_file` \
reported.
- Change as few lines as possible — this should be the smallest edit that fixes the \
behavior described in the failure, consistent with what the spec says the module should do.

`self_reported_confidence` should reflect how certain you are that this is the true root \
cause AND that you read (not guessed) the exact lines being patched, on a 0.0-1.0 scale."""


def build_synthesize_node(model: BaseChatModel):
    structured = model.with_structured_output(Hypothesis)

    def synthesize_hypothesis(state: DebugState) -> dict:
        messages = list(state["messages"]) + [HumanMessage(content=_INSTRUCTION)]
        hypothesis: Hypothesis = structured.invoke(messages)
        log_event(
            logger,
            "hypothesis",
            file=hypothesis.file,
            line=hypothesis.line,
            self_reported_confidence=hypothesis.self_reported_confidence,
        )
        return {"hypothesis": hypothesis}

    return synthesize_hypothesis
