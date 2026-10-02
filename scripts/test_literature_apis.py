#!/usr/bin/env python3
"""Safe live smoke test for CellForge literature providers.

Credentials are read exclusively from environment variables. The script never
prints tokens or raw response headers.
"""

from pathlib import Path
import argparse
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cellforge.retrieval import LiteratureRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="Adamson Perturb-seq CRISPR interference single cell",
    )
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="cellforge-retrieval-") as temp_dir:
        root = Path(temp_dir)
        retriever = LiteratureRetriever(
            literature_dir=root / "literature",
            trace_dir=root / "trace",
            online=True,
        )
        results = retriever.search(args.query, limit=args.limit)
        trace_path = root / "trace" / "retrieval_trace.jsonl"
        trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])

    print(json.dumps({
        "query": args.query,
        "providers": trace["providers"],
        "results": [
            {
                "paper_id": item["paper_id"],
                "title": item["title"],
                "source": item["source"],
                "score": item["score"],
            }
            for item in results
        ],
    }, indent=2, ensure_ascii=False))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
