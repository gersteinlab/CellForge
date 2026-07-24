#!/usr/bin/env python3
"""Copy a legacy PDF corpus into an external CellForge literature directory."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Use symlink for a zero-copy local development corpus.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    papers_dir = destination / "papers"
    manifest_dir = destination / "manifests"
    papers_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for pdf in sorted(source.glob("*.pdf")):
        target = papers_dir / pdf.name
        if not target.exists():
            if args.mode == "symlink":
                target.symlink_to(pdf)
            else:
                shutil.copy2(pdf, target)
        entries.append({
            "filename": pdf.name,
            "sha256": digest(pdf),
            "size_bytes": pdf.stat().st_size,
            "source": str(pdf),
        })

    manifest = {
        "source": str(source),
        "destination": str(destination),
        "mode": args.mode,
        "papers": entries,
    }
    target = manifest_dir / "corpus-manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Migrated {len(entries)} papers; manifest: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
