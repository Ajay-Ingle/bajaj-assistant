"""Standalone CLI to test app.funds.retrieve against the built index.

Usage: python scripts/test_rag.py "<query>" [--top-k N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.funds.retrieve import retrieve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the fund RAG retrieval layer.")
    parser.add_argument("query", help="Query string to retrieve chunks for.")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    result = retrieve(args.query, top_k=args.top_k)

    print("=" * 60)
    print(f"QUERY: {args.query}")
    print("=" * 60)
    print(f"found: {result['found']}")
    print(f"matched_fund_filter: {result.get('matched_fund_filter')}")

    if not result["found"]:
        print(f"answer_hint: {result.get('answer_hint')}")
        return

    for i, chunk in enumerate(result["chunks"], start=1):
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"--- result {i} ---")
        print(f"fund_name:        {chunk['fund_name']}")
        print(f"page_number:      {chunk['page_number']}")
        print(f"similarity_score: {chunk['similarity_score']:.4f}")
        print(f"preview:          {preview}...")
        print()


if __name__ == "__main__":
    main()
