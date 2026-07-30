#!/usr/bin/env python3
"""Package a real CellForge run into a committable example bundle.

A run leaves artifacts scattered across ``data/analyses/``, ``data/plans/`` and a
generation workspace, and those artifacts are not safe to commit as they stand:
agent traces echo API keys, and every path in them is absolute and specific to
the machine that produced them. This script collects the artifacts, scrubs them,
and writes the directory layout that ``examples/outputs/`` expects.

    python scripts/export_example_run.py --dataset adamson_crispri \\
        --out examples/outputs/adamson_crispri

It refuses to write anything it could not scrub. If a file still looks like it
contains a credential after redaction, the export fails loudly rather than
quietly committing a secret.

Nothing here needs an API key or a GPU — it only moves and rewrites text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Text extensions get scrubbed; anything else is copied verbatim or skipped.
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".mmd", ".py", ".txt", ".yaml", ".yml", ".cfg", ".toml"}

# Files that must never end up in a bundle regardless of location.
NEVER_COPY = {
    ".env",
    "config.local.json",
    "id_rsa",
    "credentials.json",
    "token.json",
}

SKIP_SUFFIXES = {".h5ad", ".h5", ".hdf5", ".loom", ".mtx", ".npz", ".npy",
                 ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx"}

# Patterns replaced during scrubbing. Order matters: longer/more specific first.
SECRET_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"sk-ant-[A-Za-z0-9_\-]{16,}", "$ANTHROPIC_API_KEY"),
    (r"sk-proj-[A-Za-z0-9_\-]{16,}", "$OPENAI_API_KEY"),
    (r"sk-[A-Za-z0-9]{32,}", "$API_KEY"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}", "$GITHUB_TOKEN"),
    (r"\bAKIA[0-9A-Z]{16}\b", "$AWS_ACCESS_KEY_ID"),
    (r"\bAIza[0-9A-Za-z_\-]{35}\b", "$GOOGLE_API_KEY"),
    (r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}", "$JWT"),
    (r"(?i)\b(api[_\-]?key|secret|token|password)\b(\s*[:=]\s*)[\"']?[A-Za-z0-9_\-]{16,}[\"']?",
     r"\1\2$REDACTED"),
)

# Anything still matching these after scrubbing aborts the export.
TRIPWIRES: Sequence[str] = (
    r"sk-ant-[A-Za-z0-9_\-]{16,}",
    r"sk-[A-Za-z0-9]{32,}",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bAIza[0-9A-Za-z_\-]{35}\b",
)


class Scrubber:
    """Rewrites machine-specific and secret-looking text out of run artifacts."""

    def __init__(self, workspace: Path, extra: Optional[Dict[str, str]] = None) -> None:
        self.replacements: List[Tuple[str, str]] = []

        # Longest paths first, so /home/u/CellForge/data resolves before /home/u.
        raw: Dict[str, str] = {
            str(workspace.resolve()): "$WORKSPACE",
            str(REPO_ROOT): "$REPO",
            str(Path.home()): "$HOME",
            sys.executable: "python3",
        }
        try:
            raw[socket.gethostname()] = "$HOSTNAME"
        except OSError:
            pass
        for name in ("USER", "LOGNAME"):
            value = os.environ.get(name)
            if value and len(value) > 2:
                raw[value] = "$USER"
        if extra:
            raw.update(extra)

        for needle, token in sorted(raw.items(), key=lambda kv: -len(kv[0])):
            if needle and needle not in {"/", "."}:
                self.replacements.append((needle, token))

    def scrub(self, text: str) -> str:
        for needle, token in self.replacements:
            text = text.replace(needle, token)
        for pattern, token in SECRET_PATTERNS:
            text = re.sub(pattern, token, text)
        return text

    @staticmethod
    def tripwire(text: str) -> List[str]:
        return [p for p in TRIPWIRES if re.search(p, text)]


def newest(directory: Path, pattern: str) -> Optional[Path]:
    """Most recently modified file matching a glob, or None."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def copy_file(src: Path, dst: Path, scrubber: Scrubber, dry_run: bool) -> Tuple[bool, str]:
    """Copy one file, scrubbing it if it is text. Returns (copied, note)."""
    if src.name in NEVER_COPY:
        return False, "refused (never-copy list)"
    if src.suffix.lower() in SKIP_SUFFIXES:
        return False, "skipped (data/model artifact)"

    if src.suffix.lower() not in TEXT_SUFFIXES:
        if dry_run:
            return True, "would copy verbatim"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True, "copied verbatim"

    try:
        text = src.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        return False, f"skipped (unreadable: {exc})"

    cleaned = scrubber.scrub(text)
    hits = scrubber.tripwire(cleaned)
    if hits:
        raise SystemExit(
            f"\nABORT: {src} still matches credential patterns after scrubbing:\n"
            + "\n".join(f"  - {h}" for h in hits)
            + "\n\nNothing was written. Redact the file by hand, or add a pattern to "
              "SECRET_PATTERNS in this script, then re-run.\n"
        )

    scrubbed = " (scrubbed)" if cleaned != text else ""
    if dry_run:
        return True, "would copy" + scrubbed
    note = "copied" + scrubbed
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")
    return True, note


def collect(dataset: str, workspace: Path, code_dir: Optional[Path]) -> List[Tuple[Path, str]]:
    """Find the artifacts of a run. Returns (source, destination-relative) pairs."""
    found: List[Tuple[Path, str]] = []

    analyses = workspace / "data" / "analyses" / dataset
    if analyses.is_dir():
        for name in ("task_analysis_report.md", "task_analysis.json"):
            candidate = analyses / name
            if candidate.is_file():
                found.append((candidate, f"analyses/{name}"))
        for extra in sorted(analyses.glob("*.md")):
            if extra.name != "task_analysis_report.md":
                found.append((extra, f"analyses/{extra.name}"))

    plans = workspace / "data" / "plans" / dataset
    if not plans.is_dir():
        plans = workspace / "data" / "plans"
    if plans.is_dir():
        # A run stamps the filename; normalise it so the bundle is stable.
        for suffix in ("md", "json", "mmd"):
            latest = newest(plans, f"research_plan_*.{suffix}")
            if latest is not None:
                found.append((latest, f"plans/research_plan.{suffix}"))

    if code_dir is not None and code_dir.is_dir():
        for name in ("result.py", "metrics.json", "verification.json", "requirements.txt"):
            candidate = code_dir / name
            if candidate.is_file():
                found.append((candidate, f"workspace/{name}"))

    return found


PROVENANCE_STUB = """# Provenance

<!-- Complete this before opening a PR. An unedited stub will be rejected. -->

## Summary

This bundle was exported from a real CellForge run with
`scripts/export_example_run.py`.

## Run details

| Field | Value |
| --- | --- |
| Dataset | {dataset} |
| Date of run | <!-- YYYY-MM-DD --> |
| CellForge commit | {commit} |
| LLM provider / model | <!-- e.g. claude-opus-5 --> |
| Code generation backend | <!-- e.g. Codex CLI --> |
| Hardware | <!-- e.g. 1x A100 80GB --> |
| Wall-clock time | <!-- e.g. 6h --> |
| Approximate token cost | <!-- prompt / completion --> |

## Per-file provenance

| File | What it is | Real run output? |
| --- | --- | --- |
{rows}

## Metrics

<!-- If metrics.json is present, state what data produced it: the real dataset,
     a subsample, or synthetic. If it is not the full dataset, say so here in
     plain terms, and do not present the numbers as benchmark results. -->

## Known caveats

<!-- Anything a reader would be misled by if you did not say it. -->
"""


def write_provenance(out: Path, dataset: str, copied: List[str], dry_run: bool) -> None:
    target = out / "PROVENANCE.md"
    if target.exists():
        print("  PROVENANCE.md exists, leaving it alone")
        return

    commit = "unknown"
    head = REPO_ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            commit = (REPO_ROOT / ".git" / ref[5:]).read_text(encoding="utf-8").strip()[:12]
        elif ref:
            commit = ref[:12]
    except OSError:
        pass

    rows = "\n".join(f"| `{path}` | <!-- describe --> | Yes |" for path in copied)
    body = PROVENANCE_STUB.format(dataset=dataset, commit=commit, rows=rows or "| | | |")
    if dry_run:
        print("  would write PROVENANCE.md stub")
        return
    target.write_text(body, encoding="utf-8")
    print("  wrote PROVENANCE.md stub — complete it before opening a PR")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package a real CellForge run into a committable example bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name as used under data/analyses/ and data/plans/.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Destination bundle directory.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                        help="Run workspace containing data/analyses and data/plans.")
    parser.add_argument("--code-dir", type=Path, default=None,
                        help="Directory holding the generated result.py. Defaults to "
                             "the newest .cellforge_workspaces/* under --workspace.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written without writing it.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite a non-empty --out directory.")
    args = parser.parse_args(argv)

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"--workspace is not a directory: {workspace}")

    code_dir = args.code_dir
    if code_dir is None:
        pool = workspace / ".cellforge_workspaces"
        if pool.is_dir():
            candidates = [p for p in pool.iterdir() if p.is_dir()]
            if candidates:
                code_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    if code_dir is not None:
        code_dir = code_dir.expanduser().resolve()

    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.force and not args.dry_run:
        parser.error(f"--out is not empty: {out}. Pass --force to overwrite.")

    print(f"workspace : {workspace}")
    print(f"code dir  : {code_dir or '(none found)'}")
    print(f"dataset   : {args.dataset}")
    print(f"out       : {out}\n")

    artifacts = collect(args.dataset, workspace, code_dir)
    if not artifacts:
        print("No run artifacts found. Expected at least one of:")
        print(f"  {workspace}/data/analyses/{args.dataset}/task_analysis_report.md")
        print(f"  {workspace}/data/plans/{args.dataset}/research_plan_*.md")
        print(f"  {code_dir or '<code-dir>'}/result.py")
        print("\nPass --workspace / --code-dir if the run lives elsewhere.")
        return 1

    scrubber = Scrubber(workspace)
    copied: List[str] = []
    for src, relative in artifacts:
        ok, note = copy_file(src, out / relative, scrubber, args.dry_run)
        print(f"  [{'ok' if ok else '--'}] {relative:44s} {note}")
        if ok:
            copied.append(relative)

    if not copied:
        print("\nNothing was eligible for copying.")
        return 1

    write_provenance(out, args.dataset, copied, args.dry_run)

    print(f"\n{len(copied)} file(s) {'would be ' if args.dry_run else ''}exported.")
    if not args.dry_run:
        print("\nBefore committing:")
        print("  1. Complete PROVENANCE.md — an unedited stub will be rejected.")
        print("  2. Re-read every file. Scrubbing is best-effort, not a guarantee.")
        print("  3. Confirm no dataset files or model weights were included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
