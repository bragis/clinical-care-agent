"""clinical-care-agent: a small, clinical-AI agent that evaluates clinical cases.

Pillars:
  - RAG over a protocol knowledge base            (src/retrieval.py)
  - LLM tool-use / function calling               (src/llm.py, src/tools.py)
  - multi-step orchestration + human-in-the-loop  (src/agent.py)
  - structured outputs                            (src/schemas.py)
  - offline eval harness + safety/regression + CI (eval/)
"""

import json
from pathlib import Path
from typing import Any, Dict

from .retrieval import TfidfRetriever, load_protocol_chunks, EmbeddingRetriever, split_into_windows
from .tools import ToolRegistry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_patients(path: Path | str = DATA_DIR / "patients.json") -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_registry(
    patients: Dict[str, Any] | None = None,
    protocols_dir: Path | str = DATA_DIR / "protocols",
) -> ToolRegistry:
    """Default wiring: TF-IDF retriever (offline). Swap to EmbeddingRetriever for Level 2."""

    #Testing
    base = load_protocol_chunks("data/protocols")
    windowed = split_into_windows(base, max_words=80, overlap_words=15)
    print(len(base), "->", len(windowed), "chunks")  # 15 -> ~19 on this data
    # Done testing
    

    patients = patients if patients is not None else load_patients()
    chunks = load_protocol_chunks(protocols_dir)
    chunks = split_into_windows(chunks, max_words=80, overlap_words=15)   # Level 2
    #retriever = TfidfRetriever(chunks)
    retriever = EmbeddingRetriever(chunks)
    return ToolRegistry(patients=patients, retriever=retriever)
