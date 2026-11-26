#!/usr/bin/env python3
"""Run the care agent on a single patient and print the reasoning trace.

Offline (default, no key):
    python run_agent.py --patient P004

Real LLM (after setting ANTHROPIC_API_KEY):
    python run_agent.py --patient P004 --client anthropic
"""

from __future__ import annotations

import argparse
import json

from src import build_registry, load_patients    # In src/__init__.py (only two functions in that file)
from src.agent import run_agent
from src.llm import AnthropicClient, MockLLMClient


def make_client(kind: str):
    if kind == "anthropic":
        return AnthropicClient()
    return MockLLMClient()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="P001")
    ap.add_argument("--client", choices=["mock", "anthropic"], default="mock")
    args = ap.parse_args()

    patients = load_patients()
    registry = build_registry(patients)
    client = make_client(args.client)

    result = run_agent(args.patient, client, registry)

    print(f"\n=== Trace for {args.patient} ({args.client} client) ===")
    for ts in result.trace:
        thought = f"  [{ts.step}] {ts.thought}".rstrip()
        print(thought)
        print(f"      -> {ts.tool}({json.dumps(ts.tool_input)[:90]})")
    print(f"\nstopped_reason: {result.stopped_reason}")
    if result.escalations:
        print("escalations:")
        for e in result.escalations:
            print(f"  - [{e.urgency}] {e.reason}")

    print("\n=== Structured recommendation ===")
    if result.recommendation:
        print(result.recommendation.model_dump_json(indent=2))
    else:
        print("(no recommendation produced)")


if __name__ == "__main__":
    main()
