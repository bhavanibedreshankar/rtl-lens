"""The investigation loop: plan -> call tools -> observe -> repeat, bounded by
`limits.max_investigation_steps`. Runs on the cheaper `models.planning` model
since tool selection is a routine, high-frequency decision — only hypothesis
synthesis escalates to the stronger model.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.logging_config import get_logger, log_event
from agent.state import DebugState, EvidenceItem
from config.schema import LimitsConfig

logger = get_logger("nodes.investigate")

_STOP_HINT = (
    "\n\nIf you now have enough evidence to state a concrete root cause (file, line, "
    "explanation), respond in plain text WITHOUT calling another tool and summarize your "
    "conclusion — you will then be asked for a structured patch."
)


def build_investigate_node(model: BaseChatModel, tools: list[BaseTool], limits: LimitsConfig):
    bound = model.bind_tools(tools)

    def investigate(state: DebugState) -> dict:
        # Prefer the LangMem-compressed view once it exists, to conserve tokens on
        # long investigations; `messages` (full, untruncated) remains the audit trail.
        #
        # Guarded rather than trusted blindly: LangMem's summarization has been observed
        # to collapse the compressed view down to system-message(s) only — non-empty as a
        # Python list, but Anthropic extracts SystemMessages into a separate `system`
        # field, so a messages array with no other content is rejected outright as empty.
        # `_has_content` catches that case specifically, not just literal emptiness.
        def _has_content(msgs: list) -> bool:
            return any(not isinstance(m, SystemMessage) for m in msgs)

        summarized = state.get("summarized_messages")
        messages = list(summarized) if summarized and _has_content(summarized) else list(state["messages"])
        if not _has_content(messages):
            log_event(
                logger,
                "empty_message_view_fallback",
                had_summarized=bool(summarized),
                summarized_len=len(summarized) if summarized else 0,
                level=30,
            )
            messages = list(state["messages"])
        if not _has_content(messages):
            # Last resort: state["messages"] itself has no real content. Rebuild a
            # minimal prompt from the failure so the run can still make progress
            # instead of crashing outright.
            from agent.nodes.ingest_failure import _SYSTEM_PROMPT

            failure = state.get("failure")
            log_event(logger, "reconstructing_messages_from_failure", level=40)
            messages = [SystemMessage(content=_SYSTEM_PROMPT)]
            if failure is not None:
                messages.append(HumanMessage(content=f"(Context was lost — resuming.) Failure: {failure.message}"))
        steps = state.get("investigation_steps", 0)
        effective_limit = _effective_limit(state, limits)

        # On the last allowed step, force a text-only reply: letting the model emit
        # one more tool_use here would leave it unanswered when we route straight to
        # synthesize, which the Anthropic API rejects (every tool_use needs a
        # tool_result in the very next message).
        is_last_step = steps >= effective_limit - 1
        if is_last_step:
            messages = messages + [
                HumanMessage(
                    content="This is your final investigation step. Do NOT call another tool — "
                    "state your root-cause conclusion in plain text now."
                )
            ]
            response = model.invoke(messages)
        else:
            response = bound.invoke(messages)

        log_event(
            logger,
            "investigate_step",
            step=steps,
            tool_calls=[tc["name"] for tc in getattr(response, "tool_calls", [])],
            forced_final=is_last_step,
        )
        return {"messages": [response], "investigation_steps": steps + 1}

    return investigate


def build_tools_node(tools: list[BaseTool]):
    by_name = {t.name: t for t in tools}

    def run_tools(state: DebugState) -> dict:
        last: AIMessage = state["messages"][-1]
        tool_messages = []
        new_evidence = list(state.get("evidence", []))
        for call in last.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                content = f"(unknown tool: {call['name']})"
            else:
                try:
                    content = tool.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001
                    content = f"(tool error: {exc})"
            content = str(content)
            tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            new_evidence.append(
                EvidenceItem(tool=call["name"], args=call["args"], summary=content[:500])
            )
            log_event(logger, "tool_call", tool=call["name"], args=call["args"])
        return {"messages": tool_messages, "evidence": new_evidence}

    return run_tools


def _effective_limit(state: DebugState, limits: LimitsConfig) -> int:
    """Each confidence-driven retry (see score_confidence.route_after_confidence)
    grants a fresh full step budget, since `investigation_steps` never resets."""
    return limits.max_investigation_steps * (1 + state.get("confidence_retries", 0))


def route_after_investigate(state: DebugState, limits: LimitsConfig) -> str:
    last = state["messages"][-1]
    steps = state.get("investigation_steps", 0)
    if getattr(last, "tool_calls", None) and steps < _effective_limit(state, limits):
        return "tools"
    return "synthesize"
