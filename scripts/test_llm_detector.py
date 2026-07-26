"""Standalone smoke test for the LLM detector agent.

Usage:
    python scripts/test_llm_detector.py                            # default doi
    python scripts/test_llm_detector.py SciPostPhys.21.1.011
    python scripts/test_llm_detector.py arxiv_2401.12345
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peerlens.agents.llm_detector import analyze_llm_authorship, save_report, MODEL_NAME


def main():
    doi = sys.argv[1] if len(sys.argv) > 1 else "SciPostPhys.20.4.097"
    print(f"Analysing {doi} with model {MODEL_NAME}...")
    rep = analyze_llm_authorship(doi)
    path = save_report(rep)
    print()
    print(f"Verdict:    {rep.verdict}")
    print(f"Confidence: {rep.confidence:.2f}")
    print(f"Signals:    {rep.n_signals} total "
          f"(strong={rep.n_strong}, moderate={rep.n_moderate}, weak={rep.n_weak})")
    print(f"Reasoning:  {rep.reasoning}")
    print()
    if rep.signals:
        print("Individual signals:")
        for s in rep.signals:
            print(f"  [{s.get('severity','?'):<8}] {s.get('category','?'):<18} "
                  f"@ {s.get('location','?')}: {s.get('description','')[:130]}")
        print()
    # Cost estimate for gemini-flash-latest (approx: $0.10/M input, $0.40/M output)
    cost = (rep.input_tokens / 1e6) * 0.10 + (rep.output_tokens / 1e6) * 0.40
    print(f"Tokens: input={rep.input_tokens}, output={rep.output_tokens}   cost ~${cost:.4f}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
