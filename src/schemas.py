"""Structured output schemas.

We force the agent's final answer
through a typed schema (via a `submit_recommendation` tool) so the output is
machine-checkable rather than free text. This is what makes the eval harness
able to assert on `escalate`, `citations`, etc.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A grounding reference back into the retrieved protocol knowledge base."""

    source: str = Field(..., description="Protocol document id, e.g. 'hypertension'.")
    snippet: str = Field(..., description="The retrieved text the claim is grounded in.")


class CareRecommendation(BaseModel):
    """The agent's final, structured care-review output.

    IMPORTANT: this is a DRAFT for clinician review, never an autonomous order.
    """

    patient_id: str
    summary: str = Field(..., description="One-paragraph clinical-state summary.")
    assessment: str = Field(..., description="Reasoning over vitals/labs vs. protocol.")
    draft_recommendations: List[str] = Field(
        default_factory=list,
        description="Suggested next steps phrased as options for the clinician.",
    )
    escalate: bool = Field(
        ...,
        description="True if protocol escalation criteria are met and a clinician "
        "must review same-day / urgently.",
    )
    escalation_reason: Optional[str] = Field(
        None, description="Why escalation was triggered (required if escalate=True)."
    )
    urgency: Literal["routine", "same_day", "urgent"] = "routine"
    citations: List[Citation] = Field(
        default_factory=list,
        description="Protocol grounding for the assessment and recommendations.",
    )


# JSON schema handed to the LLM for the `submit_recommendation` tool.
# Kept in sync with CareRecommendation by deriving from the pydantic model.
def recommendation_tool_schema() -> dict:
    schema = CareRecommendation.model_json_schema()
    # Anthropic/OpenAI tool input schemas are plain JSON Schema objects.
    return schema
