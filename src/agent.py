"""Agent orchestration loop.

The loop is deliberately small and readable:

    user request
      -> model plans + calls a tool
      -> we execute the tool, append the result
      -> repeat until the model calls `submit_recommendation` (terminal)
         or we hit the step budget

Error recovery: tool exceptions/validation errors are caught and returned to
the model as `tool_result` error payloads so it can retry, instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm import LLMClient
from .schemas import CareRecommendation
from .tools import Escalation, ToolRegistry, tool_result_to_str

SYSTEM_PROMPT = """\
You are a clinical-support agent for a chronic-disease care team (hypertension,
heart failure, diabetes). You prepare a DRAFT care review for a human clinician.

Hard rules:
- You do NOT make autonomous clinical decisions. Every recommendation is a draft
  option for a clinician, who remains in control. Phrase suggestions accordingly.
- Ground every assessment and recommendation in the protocol knowledge base by
  calling `search_protocols`, and cite what you used.
- If protocol escalation criteria are met (e.g., hypertensive-crisis BP, low SpO2
  with weight gain), call `escalate_to_clinician` before finishing.
- Never invent a specific drug or dose that is not supported by a retrieved
  protocol passage. Do not recommend starting or dosing insulin yourself.
- Finish by calling `submit_recommendation` with the structured review.

Workflow: get_patient -> get_vitals -> search_protocols -> (escalate if needed)
-> submit_recommendation.
"""


@dataclass
class TraceStep:
    step: int
    thought: str
    tool: str
    tool_input: Dict[str, Any]
    tool_result: Any


@dataclass
class AgentResult:
    patient_id: str
    recommendation: Optional[CareRecommendation]
    escalations: List[Escalation] = field(default_factory=list)
    trace: List[TraceStep] = field(default_factory=list)
    steps: int = 0
    stopped_reason: str = ""


def run_agent(
    patient_id: str,
    client: LLMClient,
    registry: ToolRegistry,
    max_steps: int = 8,
) -> AgentResult:
    tools = registry.specs()
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": f"Review patient {patient_id} and prepare a care review."}
    ]
    trace: List[TraceStep] = []
    stopped_reason = "step_budget_exhausted"

    for step in range(1, max_steps + 1):
        turn = client.create(SYSTEM_PROMPT, messages, tools)
        messages.append({"role": "assistant", "content": turn.assistant_content})

        if not turn.tool_calls:
            # Model returned plain text without finishing. Nudge once, else stop.
            stopped_reason = "model_returned_no_tool_call"
            break

        tool_result_blocks: List[Dict[str, Any]] = []
        terminal = False
        for call in turn.tool_calls:
            try:
                result, is_terminal = registry.execute(call.name, call.input)
            except Exception as e:  # error recovery: surface to model, keep going
                result, is_terminal = {"error": f"tool raised: {e}"}, False
            terminal = terminal or is_terminal
            trace.append(TraceStep(step, turn.text, call.name, call.input, result))
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": tool_result_to_str(result),
            })

        messages.append({"role": "user", "content": tool_result_blocks})

        if terminal:
            stopped_reason = "submitted"
            break
    else:
        stopped_reason = "step_budget_exhausted"

    return AgentResult(
        patient_id=patient_id,
        recommendation=registry.final,
        escalations=registry.escalations,
        trace=trace,
        steps=len(trace),
        stopped_reason=stopped_reason,
    )
