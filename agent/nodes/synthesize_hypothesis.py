from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agent.logging_config import get_logger, log_event
from agent.state import DebugState, Hypothesis, HypothesisDraft
from agent.tools.rtl_tools import DiffMismatchError, ScopeError, build_unified_diff
from config.schema import RtlRepoConfig

logger = get_logger("nodes.synthesize_hypothesis")

_INSTRUCTION = """Based on your investigation above, produce your final diagnosis.

CRITICAL: base `old_lines` ONLY on source text you actually saw via `rtl_read_file` in this \
conversation. If you have not read the exact lines you're about to change, call \
`rtl_read_file` again before answering — do not reconstruct or guess them from the module's \
general shape or from what a "typical" implementation might look like. `old_lines` must be a \
verbatim, line-for-line copy of what's actually in the file (no line-number prefixes) — it is \
checked against the real source before your patch is ever shown to a reviewer, and a mismatch \
means your diagnosis wasn't actually grounded in what you read.

- Keep the edit minimal: `old_lines`/`new_lines` should cover only the lines that actually \
change, not the whole surrounding block — this should be the smallest edit that fixes the \
behavior described in the failure, consistent with what the spec says the module should do.
- `self_reported_confidence` should reflect how certain you are that this is the true root \
cause AND that `old_lines` is a verbatim copy of what you read, on a 0.0-1.0 scale.
- If you genuinely don't have enough evidence yet: still give your single best-guess file, \
line, and a verbatim old_lines/new_lines pair (never placeholder text in those fields — an \
empty list is fine if you have nothing), and set self_reported_confidence very low (e.g. 0.05). \
Never write filler like "PLACEHOLDER" or "TBD" into old_lines/new_lines.

FORMAT: `old_lines` and `new_lines` are JSON arrays of strings — each array element is exactly \
ONE physical source line. If the code you're changing spans multiple lines (e.g. a multi-line \
ternary or a wrapped expression), that is multiple array elements, one per line, in order — \
never join several lines into a single string with embedded newlines."""

_RETRY_INSTRUCTION = """Your previous response didn't match the required schema: {error}

Remember: `old_lines`/`new_lines` are JSON arrays where each element is exactly one physical \
source line — never a single string containing multiple lines joined by newlines. Please retry."""


def build_synthesize_node(model: BaseChatModel, rtl_repo_cfg: RtlRepoConfig):
    structured = model.with_structured_output(HypothesisDraft)

    def synthesize_hypothesis(state: DebugState) -> dict:
        messages = list(state["messages"]) + [HumanMessage(content=_INSTRUCTION)]
        draft: HypothesisDraft | None = None
        last_error: Exception | None = None
        # One retry for malformed structured output (e.g. multi-line strings crammed into a
        # single old_lines/new_lines element): worth a second attempt with the concrete error
        # before giving up and forcing a full extra investigation pass over it.
        for attempt_messages in (messages, None):
            if attempt_messages is None:
                if last_error is None:
                    break
                attempt_messages = messages + [
                    HumanMessage(content=_RETRY_INSTRUCTION.format(error=last_error))
                ]
            try:
                draft = structured.invoke(attempt_messages)
                break
            except Exception as exc:  # noqa: BLE001 — malformed structured output
                last_error = exc
                log_event(logger, "structured_output_failed", error=str(exc), level=30)

        if draft is None:
            hypothesis = Hypothesis(
                file="(unknown)",
                line=None,
                explanation=f"Failed to produce a valid structured hypothesis after retry: {last_error}",
                patch_diff="",
                self_reported_confidence=0.0,
            )
            return {"hypothesis": hypothesis}

        try:
            patch_diff = build_unified_diff(
                rtl_repo_cfg, draft.file, draft.start_line, draft.old_lines, draft.new_lines
            )
            confidence = draft.self_reported_confidence
        except (DiffMismatchError, ScopeError) as exc:
            log_event(logger, "diff_mismatch", file=draft.file, line=draft.start_line, error=str(exc), level=30)
            patch_diff = ""
            # Ungrounded hypothesis: force low confidence regardless of what the model
            # claimed, so score_confidence routes back for another investigation pass.
            confidence = 0.0
            draft.explanation += (
                f"\n\n[Diagnosis error: the claimed source lines at {draft.file}:{draft.start_line} "
                "did not match the actual file — this hypothesis was not grounded in what was read.]"
            )

        hypothesis = Hypothesis(
            file=draft.file,
            line=draft.start_line,
            explanation=draft.explanation,
            patch_diff=patch_diff,
            self_reported_confidence=confidence,
        )
        log_event(
            logger,
            "hypothesis",
            file=hypothesis.file,
            line=hypothesis.line,
            self_reported_confidence=hypothesis.self_reported_confidence,
        )
        return {"hypothesis": hypothesis}

    return synthesize_hypothesis
