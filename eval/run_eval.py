#!/usr/bin/env python3
"""Run the evaluation suite over all cases and print a report.

Offline (deterministic, no key) -- this is what CI runs:
    python eval/run_eval.py

With LLM-as-judge (needs ANTHROPIC_API_KEY):
    python eval/run_eval.py --client anthropic --judge

Exit code is non-zero if any SAFETY or REGRESSION case fails, so it can gate a
pull request. Behavioural-only failures warn but do not fail the build (tune to
taste).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge import CaseScore, deterministic_checks, llm_judge  # noqa: E402
from src import build_registry, load_patients  # noqa: E402
from src.agent import run_agent  # noqa: E402
from src.llm import AnthropicClient, MockLLMClient  # noqa: E402

GATING_KINDS = {"safety", "regression"}


def make_client(kind: str):
    return AnthropicClient() if kind == "anthropic" else MockLLMClient()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--judge", action="store_true", help="run LLM-as-judge (needs a real client)")
    ap.add_argument("--cases", default=str(Path(__file__).resolve().parent / "cases.yaml"))
    args = ap.parse_args()

    cases = yaml.safe_load(Path(args.cases).read_text())["cases"]
    patients = load_patients()
    judge_client = AnthropicClient() if (args.judge and args.client == "anthropic") else None

    scores: list[CaseScore] = []
    for case in cases:
        registry = build_registry(patients)        # fresh registry per case
        client = make_client(args.client)
        result = run_agent(case["patient_id"], client, registry)
        rec = result.recommendation

        cs = CaseScore(case_id=case["id"], kind=case["kind"])
        cs.checks = deterministic_checks(case, rec)
        if judge_client is not None and rec is not None and case.get("judge_rubric"):
            cs.judge_score = llm_judge(judge_client, case["judge_rubric"], rec)
        scores.append(cs)

    # -- report --
    print(f"\nEvaluation report  (client={args.client}, judge={'on' if judge_client else 'off'})")
    print("=" * 72)
    gate_failed = False
    for cs in scores:
        status = "PASS" if cs.passed else "FAIL"
        judge = f"  judge={cs.judge_score:.2f}" if cs.judge_score is not None else ""
        print(f"[{status}] {cs.case_id:<32} ({cs.kind}){judge}")
        for c in cs.checks:
            if not c.passed:
                print(f"        x {c.name}: {c.detail}")
        if not cs.passed and cs.kind in GATING_KINDS:
            gate_failed = True

    n_pass = sum(cs.passed for cs in scores)
    print("-" * 72)
    print(f"{n_pass}/{len(scores)} cases passed deterministic checks.")
    if gate_failed:
        print("GATE FAILED: a safety/regression case failed.")
        return 1
    print("Gate OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
