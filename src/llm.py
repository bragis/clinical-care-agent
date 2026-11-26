"""LLM client abstraction.


The agent loop is provider-agnostic: it talks to any object implementing
`LLMClient.create(...)`, which returns an `AssistantTurn`. Two implementations:

* `AnthropicClient`  -- the real Messages API tool-use call.
* `MockLLMClient`    -- a deterministic, scripted "agent" that follows a fixed
                        tool-use policy by reading the conversation history.
                        No API key, no network. This is what lets the eval
                        harness and CI run reproducibly offline.

Swapping mock -> real is a one-line change in run_agent.py / run_eval.py.

"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class AssistantTurn:
    text: str
    tool_calls: List[ToolCall]
    stop_reason: str
    # Provider-native assistant content blocks, to append back into history.
    assistant_content: List[Dict[str, Any]] = field(default_factory=list)


class LLMClient(Protocol):
    def create(
        self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> AssistantTurn: ...


# ---------------------------------------------------------------------------
# Real Anthropic client
# ---------------------------------------------------------------------------
class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic to use AnthropicClient") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it, "
                "or use MockLLMClient for the offline path."
            )
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def create(self, system, messages, tools) -> AssistantTurn:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        assistant_content: List[Dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
                assistant_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
        return AssistantTurn(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            assistant_content=assistant_content,
        )


# ---------------------------------------------------------------------------
# Mock client: a deterministic scripted agent (offline)
# ---------------------------------------------------------------------------
class MockLLMClient:
    """Simulates an agent by following a fixed policy derived from history.

    Policy (each call emits exactly one tool use):
      1. get_patient            (if not yet called)
      2. get_vitals             (if not yet called)
      3. search_protocols       (if not yet called)
      4. escalate_to_clinician  (only if red-flag vitals and not yet escalated)
      5. submit_recommendation  (terminal)

    It reads prior tool_result blocks out of `messages` so it can fill the final
    structured recommendation from the data it "observed" -- staying stateless.
    """

    _next_id = 0

    def _mk_id(self) -> str:
        MockLLMClient._next_id += 1
        return f"mock_tool_{MockLLMClient._next_id}"

    @staticmethod
    def _collect_tool_results(messages) -> Dict[str, Any]:
        """Walk history, map tool_use id->name, then attach parsed results by name."""
        id_to_name: Dict[str, str] = {}
        results: Dict[str, Any] = {}
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "tool_use":
                    id_to_name[block["id"]] = block["name"]
                elif block.get("type") == "tool_result":
                    name = id_to_name.get(block.get("tool_use_id"), "")
                    raw = block.get("content")
                    try:
                        results[name] = json.loads(raw) if isinstance(raw, str) else raw
                    except (json.JSONDecodeError, TypeError):
                        results[name] = raw
        return results

    @staticmethod
    def _red_flags(vitals: dict) -> Optional[str]:
        """Return an escalation reason if the latest vitals trip protocol thresholds."""
        series = vitals.get("vitals", []) if isinstance(vitals, dict) else []
        if not series:
            return None
        latest = series[-1]
        sbp, dbp = latest.get("sbp", 0), latest.get("dbp", 0)
        spo2 = latest.get("spo2", 100)
        if sbp >= 180 or dbp >= 110:
            return f"Hypertensive crisis range (BP {sbp}/{dbp} mmHg)."
        if spo2 < 92:
            # Check weight trend for HF decompensation signal.
            weights = [v.get("weight_kg") for v in series if v.get("weight_kg")]
            gained = len(weights) >= 2 and (weights[-1] - weights[0]) >= 2.0
            trend = " with rising weight" if gained else ""
            return f"Low oxygen saturation (SpO2 {spo2}%){trend}."
        return None

    @staticmethod
    def _lab_red_flags(labs):
        # latest_k = max((l for l in labs if l.get("name") == "potassium"),
        #                key=lambda l: l.get("date", ""), default=None)
        # if latest_k and latest_k.get("value", 0) >= 6.0:
        #     return f"Severe hyperkalemia (K {latest_k['value']} mmol/L)."
        # return None
        k = [l for l in labs if l.get("name") == "potassium"]
        if not k:
            return None
        latest_date = max(l.get("date", "") for l in k)
        worst = max(l.get("value", 0) for l in k if l.get("date", "") == latest_date)
        if worst >= 6.0:
            return f"Severe hyperkalemia (K {worst} mmol/L)."
        return None

    def create(self, system, messages, tools) -> AssistantTurn:
        results = self._collect_tool_results(messages)
        called = set(results.keys())

        def emit(thought: str, name: str, tool_input: dict) -> AssistantTurn:
            tid = self._mk_id()
            return AssistantTurn(
                text=thought,
                tool_calls=[ToolCall(id=tid, name=name, input=tool_input)],
                stop_reason="tool_use",
                assistant_content=[
                    {"type": "text", "text": thought},
                    {"type": "tool_use", "id": tid, "name": name, "input": tool_input},
                ],
            )

        patient = results.get("get_patient", {}) or {}
        patient_id = patient.get("patient_id", self._infer_patient_id(messages))
        conditions = patient.get("conditions", [])

        if "get_patient" not in called:
            return emit("Pulling patient context.", "get_patient", {"patient_id": patient_id})
        if "get_vitals" not in called:
            return emit("Reviewing recent vitals.", "get_vitals", {"patient_id": patient_id})
        if "search_protocols" not in called:
            query = " ".join(conditions) + " escalation criteria titration monitoring"
            return emit("Grounding against protocols.", "search_protocols",
                        {"query": query.strip(), "k": 3})

        vitals = results.get("get_vitals", {}) or {}
        # reason = self._red_flags(vitals)
        reason = self._red_flags(vitals) or self._lab_red_flags(patient.get("labs", []))
        if reason and "escalate_to_clinician" not in called:
            urgency = "same_day"
            return emit(f"Red-flag detected: {reason} Escalating.",
                        "escalate_to_clinician", {"reason": reason, "urgency": urgency})

        # Terminal: assemble a structured recommendation grounded in retrieval.
        retrieved = results.get("search_protocols", []) or []
        # A real model cites protocols matching the patient's conditions; mirror that
        # so grounding is condition-relevant rather than whatever TF-IDF ranked first.
        relevant_sources = self._sources_for_conditions(conditions)
        relevant = [c for c in retrieved if c.get("source") in relevant_sources] or retrieved
        citations = [
            {"source": c.get("source", "protocol"),
             "snippet": (c.get("text", "")[:160].replace("\n", " "))}
            for c in relevant[:2]
        ]
        escalate = reason is not None
        latest = (vitals.get("vitals") or [{}])[-1]
        summary = (
            f"{patient.get('age','?')}yo {patient.get('sex','?')} with "
            f"{', '.join(conditions) or 'no listed conditions'}; "
            f"latest BP {latest.get('sbp','?')}/{latest.get('dbp','?')}, "
            f"SpO2 {latest.get('spo2','?')}%."
        )
        rec = {
            "patient_id": patient_id,
            "summary": summary,
            "assessment": (
                "Escalation criteria met per protocol; deferring management to clinician."
                if escalate else
                "No escalation criteria met. Values reviewed against protocol targets; "
                "any medication change is a clinician decision."
            ),
            "draft_recommendations": (
                ["Flagged for same-day clinician review (see escalation)."]
                if escalate else
                ["Continue current regimen.",
                 "Clinician to consider protocol-directed titration if above target.",
                 "Reinforce home monitoring and follow-up."]
            ),
            "escalate": escalate,
            "escalation_reason": reason,
            "urgency": "same_day" if escalate else "routine",
            "citations": citations,
        }
        return emit("Submitting structured recommendation for clinician review.",
                    "submit_recommendation", rec)

    @staticmethod
    def _sources_for_conditions(conditions) -> set:
        """Map free-text conditions to protocol document ids for citation."""
        srcs = set()
        for c in conditions:
            cl = c.lower()
            if "hypertension" in cl:
                srcs.add("hypertension")
            if "diabetes" in cl:
                srcs.add("diabetes")
            if "heart failure" in cl or "hfref" in cl:
                srcs.add("heart_failure")
            if "kidney" in cl or "ckd" in cl:
                srcs.add("ckd")
        return srcs

    @staticmethod
    def _infer_patient_id(messages) -> str:
        for msg in messages:
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            m = re.search(r"\bP\d{3}\b", text)
            if m:
                return m.group(0)
        return "P001"
