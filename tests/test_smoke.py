"""Smoke tests: the whole pipeline runs offline and behaves on key cases.

Run: pytest -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_registry, load_patients
from src.agent import run_agent
from src.llm import MockLLMClient
from src.retrieval import TfidfRetriever, load_protocol_chunks


def test_retriever_returns_relevant_protocol():
    chunks = load_protocol_chunks(Path(__file__).resolve().parent.parent / "data" / "protocols")
    r = TfidfRetriever(chunks)
    hits = r.search("blood pressure hypertensive crisis escalation", k=3)
    assert hits, "expected at least one retrieved chunk"
    assert hits[0].source == "hypertension"


def test_pipeline_runs_end_to_end():
    registry = build_registry(load_patients())
    result = run_agent("P001", MockLLMClient(), registry)
    assert result.recommendation is not None
    assert result.recommendation.patient_id == "P001"
    assert result.stopped_reason == "submitted"
    # Grounding: a recommendation should carry citations.
    assert len(result.recommendation.citations) > 0


def test_hypertensive_crisis_triggers_escalation():
    registry = build_registry(load_patients())
    result = run_agent("P004", MockLLMClient(), registry)  # SBP 188
    assert result.recommendation.escalate is True
    assert result.escalations, "expected an escalation to be recorded"


def test_stable_patient_does_not_escalate():
    registry = build_registry(load_patients())
    result = run_agent("P003", MockLLMClient(), registry)  # at target
    assert result.recommendation.escalate is False
    assert not result.escalations
