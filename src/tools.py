"""Tools the agent can call.


A `ToolRegistry` owns the patient data + retriever and exposes:
  * `specs()`            -> JSON-schema tool definitions handed to the LLM
  * `execute(name, inp)` -> runs a tool, returns (result, is_terminal)

`submit_recommendation` is the terminal tool: executing it validates the
structured output and ends the agent loop. `escalate_to_clinician` is the
human-in-the-loop side effect -- it records an escalation that, in a real
system, would page the on-call clinician.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .retrieval import Retriever
from .schemas import CareRecommendation, recommendation_tool_schema


@dataclass
class Escalation:
    reason: str
    urgency: str


class ToolRegistry:
    def __init__(self, patients: Dict[str, Any], retriever: Retriever):
        self.patients = patients
        self.retriever = retriever
        self.escalations: List[Escalation] = []
        self.final: CareRecommendation | None = None

    # -- tool schemas advertised to the model --------------------------------
    def specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_patient",
                "description": "Get demographics, conditions, current medications, and labs "
                               "for a patient by id.",
                "input_schema": {
                    "type": "object",
                    "properties": {"patient_id": {"type": "string"}},
                    "required": ["patient_id"],
                },
            },
            {
                "name": "get_vitals",
                "description": "Get the recent vital-sign time series (BP, HR, SpO2, weight) "
                               "for a patient by id.",
                "input_schema": {
                    "type": "object",
                    "properties": {"patient_id": {"type": "string"}},
                    "required": ["patient_id"],
                },
            },
            {
                "name": "search_protocols",
                "description": "Retrieve relevant passages from the clinical protocol knowledge "
                               "base. Use this to ground every assessment and recommendation. "
                               "Returns passages with their source id for citation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "escalate_to_clinician",
                "description": "Escalate to a human clinician for same-day/urgent review when "
                               "protocol escalation criteria are met. The clinician remains in "
                               "control of all clinical decisions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "urgency": {"type": "string", "enum": ["same_day", "urgent"]},
                    },
                    "required": ["reason", "urgency"],
                },
            },
            {
                "name": "submit_recommendation",
                "description": "Submit the final structured care review for clinician sign-off. "
                               "This ends the review. Recommendations are DRAFTS for a clinician, "
                               "not orders.",
                "input_schema": recommendation_tool_schema(),
            },
        ]

    # -- execution -----------------------------------------------------------
    def execute(self, name: str, tool_input: Dict[str, Any]) -> Tuple[Any, bool]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"unknown tool '{name}'"}, False
        return handler(tool_input)

    def _tool_get_patient(self, inp) -> Tuple[Any, bool]:
        p = self.patients.get(inp.get("patient_id"))
        if not p:
            return {"error": "patient not found"}, False
        # Return everything except the raw vitals series (that's get_vitals' job).
        return {k: v for k, v in p.items() if k != "vitals"}, False

    def _tool_get_vitals(self, inp) -> Tuple[Any, bool]:
        p = self.patients.get(inp.get("patient_id"))
        if not p:
            return {"error": "patient not found"}, False
        return {"patient_id": p["patient_id"], "vitals": p.get("vitals", [])}, False

    def _tool_search_protocols(self, inp) -> Tuple[Any, bool]:
        k = int(inp.get("k", 3))
        hits = self.retriever.search(inp.get("query", ""), k=k)
        return [{"source": h.source, "text": h.text, "score": round(h.score, 4)} for h in hits], False

    def _tool_escalate_to_clinician(self, inp) -> Tuple[Any, bool]:
        esc = Escalation(reason=inp.get("reason", ""), urgency=inp.get("urgency", "same_day"))
        self.escalations.append(esc)
        return {"status": "escalation_recorded", "urgency": esc.urgency}, False

    def _tool_submit_recommendation(self, inp) -> Tuple[Any, bool]:
        try:
            rec = CareRecommendation.model_validate(inp)
        except Exception as e:  # validation error -> let the model retry
            return {"error": f"schema validation failed: {e}"}, False
        self.final = rec
        return {"status": "recommendation_accepted"}, True  # terminal


def tool_result_to_str(result: Any) -> str:
    """Tool results go back to the model as strings in a tool_result block."""
    return result if isinstance(result, str) else json.dumps(result)
