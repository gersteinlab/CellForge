#!/usr/bin/env python3
"""Collect autorun score trajectories from task ``run_state.json`` files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def iter_states(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("run_state.json"))


def row_from_entry(state: dict[str, Any], state_path: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    if "iter" not in entry:
        return None
    metrics = entry.get("metrics") or {}
    return {
        "dataset": state.get("dataset_name") or state_path.parent.parent.name,
        "state_path": str(state_path),
        "iteration": entry.get("iter"),
        "worker": entry.get("worker"),
        "hypothesis": entry.get("hypothesis", ""),
        "run_ok": entry.get("run_ok"),
        "job_state": entry.get("job_state", ""),
        "val_pearson": metrics.get("val_pearson", ""),
        "peak_vram_mb": metrics.get("peak_vram_mb", ""),
        "improved": entry.get("improved"),
        "best_score_after_run": "",
        "stdout_path": entry.get("stdout_path", ""),
        "stderr_path": entry.get("stderr_path", ""),
    }


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_path in iter_states(root):
        with state_path.open() as f:
            state = json.load(f)
        best = float("-inf")
        for entry in state.get("history", []):
            row = row_from_entry(state, state_path, entry)
            if row is None:
                continue
            score = row["val_pearson"]
            if isinstance(score, (int, float)) and score > best:
                best = float(score)
            row["best_score_after_run"] = "" if best == float("-inf") else best
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/runs"),
        help="Autorun directory or a single run_state.json path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/score_paths.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "state_path",
        "iteration",
        "worker",
        "hypothesis",
        "run_ok",
        "job_state",
        "val_pearson",
        "peak_vram_mb",
        "improved",
        "best_score_after_run",
        "stdout_path",
        "stderr_path",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} score rows to {args.out}")
    for row in rows:
        print(
            f"{row['dataset']} iter={row['iteration']} "
            f"val_pearson={row['val_pearson']} best={row['best_score_after_run']}"
        )


if __name__ == "__main__":
    main()
