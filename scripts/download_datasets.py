#!/usr/bin/env python3
"""Download the CellForge benchmark datasets from scPerturb.

The six datasets used in the paper are distributed as preprocessed AnnData files
by scPerturb, archived on Zenodo:

  RNA / protein   10.5281/zenodo.13350497
  ATAC            10.5281/zenodo.7058382

Use these rather than reprocessing from GEO. They are what the paper used, and
reprocessing is a large source of irreproducibility.

Examples
--------
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py adamson
    python scripts/download_datasets.py norman papalexi --out data/datasets/
    python scripts/download_datasets.py --all --dry-run

Files already present with a matching checksum are skipped, so re-running is
cheap and interrupted downloads resume by re-fetching only what failed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ZENODO_RNA = "13350497"
ZENODO_ATAC = "7058382"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    record: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return f"https://zenodo.org/records/{self.record}/files/{self.name}?download=1"


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    perturbation: str
    modality: str
    accession: str
    files: tuple = field(default_factory=tuple)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


DATASETS = {
    "adamson": Dataset(
        key="adamson",
        title="Adamson et al. 2016",
        perturbation="CRISPRi (single gene)",
        modality="scRNA-seq",
        accession="GSE90546",
        files=(
            RemoteFile(
                "AdamsonWeissman2016_GSM2406675_10X001.h5ad",
                ZENODO_RNA,
                34557246,
                "232f7e3756d41602bbe434b50662a76f",
            ),
            RemoteFile(
                "AdamsonWeissman2016_GSM2406677_10X005.h5ad",
                ZENODO_RNA,
                139059637,
                "8657391920e7f8b3e6fd52745777002a",
            ),
            RemoteFile(
                "AdamsonWeissman2016_GSM2406681_10X010.h5ad",
                ZENODO_RNA,
                471286951,
                "2fa44ea61a8dd35742af618638ec65fc",
            ),
        ),
    ),
    "norman": Dataset(
        key="norman",
        title="Norman et al. 2019",
        perturbation="CRISPRa (single + combinatorial)",
        modality="scRNA-seq",
        accession="GSE133344",
        files=(
            RemoteFile(
                "NormanWeissman2019_filtered.h5ad",
                ZENODO_RNA,
                698680199,
                "c870e6967d91c017d9da827bab183cd6",
            ),
        ),
    ),
    "srivatsan": Dataset(
        key="srivatsan",
        title="Srivatsan et al. 2020 (sci-Plex 3)",
        perturbation="small molecules, multiple doses",
        modality="scRNA-seq",
        accession="GSE139944",
        files=(
            RemoteFile(
                "SrivatsanTrapnell2020_sciplex3.h5ad",
                ZENODO_RNA,
                2526631614,
                "c9e70629505d98c7ca1a837f62b14e89",
            ),
        ),
    ),
    "schiebinger": Dataset(
        key="schiebinger",
        title="Schiebinger et al. 2019",
        perturbation="cytokine stimulation, time course",
        modality="scRNA-seq",
        accession="GSE106340",
        files=(
            RemoteFile(
                "SchiebingerLander2019_GSE106340.h5ad",
                ZENODO_RNA,
                378858726,
                "8d988c345930195ea6e64a05c762abb3",
            ),
        ),
    ),
    "papalexi": Dataset(
        key="papalexi",
        title="Papalexi et al. 2021 (ECCITE-seq)",
        perturbation="CRISPR",
        modality="CITE-seq (RNA + 200 proteins)",
        accession="GSE153056",
        files=(
            RemoteFile(
                "PapalexiSatija2021_eccite_RNA.h5ad",
                ZENODO_RNA,
                147215278,
                "d9e8bfe20b0e7b0919be042e8fcf6d03",
            ),
            RemoteFile(
                "PapalexiSatija2021_eccite_protein.h5ad",
                ZENODO_RNA,
                1191551,
                "07290242b0e835c9474bb816de9cda45",
            ),
        ),
    ),
    "liscovitch": Dataset(
        key="liscovitch",
        title="Liscovitch-Brauer et al. 2021",
        perturbation="CRISPR",
        modality="scATAC-seq",
        accession="GSE168851",
        files=(
            RemoteFile(
                "Liscovitch-BrauerSanjana2021_K562_1.zip",
                ZENODO_ATAC,
                202476643,
                "43c1ac2e1cfc0294a79ecc8c452726b5",
            ),
            RemoteFile(
                "Liscovitch-BrauerSanjana2021_K562_2.zip",
                ZENODO_ATAC,
                347307705,
                "4cadbeec133fa071f3aeffcabdf71867",
            ),
        ),
    ),
}

ALIASES = {
    "adamson2016": "adamson",
    "norman2019": "norman",
    "sciplex": "srivatsan",
    "sciplex3": "srivatsan",
    "srivatsan2020": "srivatsan",
    "schiebinger2019": "schiebinger",
    "papalexi2021": "papalexi",
    "eccite": "papalexi",
    "atac": "liscovitch",
    "liscovitch-brauer": "liscovitch",
}


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching Zenodo's published checksum
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def print_catalogue() -> None:
    print(f"\n  CellForge benchmark datasets (scPerturb)\n  {'-' * 74}")
    for ds in DATASETS.values():
        print(f"  {ds.key:<14} {ds.title}")
        print(f"  {'':<14} {ds.perturbation} · {ds.modality}")
        print(f"  {'':<14} {ds.accession} · {len(ds.files)} file(s) · {human(ds.total_bytes)}")
        print()
    total = sum(d.total_bytes for d in DATASETS.values())
    print(f"  {'-' * 74}\n  All six: {human(total)}\n")
    print("  RNA/protein: https://doi.org/10.5281/zenodo.13350497")
    print("  ATAC:        https://doi.org/10.5281/zenodo.7058382\n")


def _progress(done: int, total: int, name: str) -> None:
    if not sys.stderr.isatty():
        return
    if total <= 0:
        sys.stderr.write(f"\r    {name}: {human(done)}")
    else:
        pct = 100 * done / total
        filled = int(pct // 4)
        bar = "█" * filled + "░" * (25 - filled)
        sys.stderr.write(f"\r    {bar} {pct:5.1f}%  {human(done)} / {human(total)}")
    sys.stderr.flush()


def download(remote: RemoteFile, dest: Path) -> bool:
    """Fetch one file, verifying its checksum. Returns True on success."""
    if dest.exists():
        print(f"    verifying existing {dest.name} ...", end=" ", flush=True)
        if md5sum(dest) == remote.md5:
            print("ok, skipping")
            return True
        print("checksum mismatch, re-downloading")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"    {dest.name}  ({human(remote.size)})")
    try:
        request = urllib.request.Request(remote.url, headers={"User-Agent": "cellforge-downloader"})
        with urllib.request.urlopen(request) as response, tmp.open("wb") as handle:  # noqa: S310
            total = int(response.headers.get("Content-Length") or remote.size)
            done = 0
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                _progress(done, total, dest.name)
        if sys.stderr.isatty():
            sys.stderr.write("\n")
    except (urllib.error.URLError, OSError) as exc:
        if sys.stderr.isatty():
            sys.stderr.write("\n")
        print(f"    ✗ download failed: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    print("    verifying checksum ...", end=" ", flush=True)
    if md5sum(tmp) != remote.md5:
        print("FAILED")
        print(
            f"    ✗ {dest.name} is corrupt (expected md5 {remote.md5}). Removed.",
            file=sys.stderr,
        )
        tmp.unlink(missing_ok=True)
        return False
    print("ok")

    tmp.replace(dest)
    return True


def resolve(names) -> list:
    resolved, unknown = [], []
    for raw in names:
        key = ALIASES.get(raw.lower(), raw.lower())
        if key in DATASETS:
            if key not in resolved:
                resolved.append(key)
        else:
            unknown.append(raw)
    if unknown:
        print(f"Unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(DATASETS)}", file=sys.stderr)
        print("Run with --list for details.", file=sys.stderr)
        raise SystemExit(2)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download CellForge benchmark datasets from scPerturb.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Datasets: " + ", ".join(DATASETS),
    )
    parser.add_argument("datasets", nargs="*", help="Dataset keys to download")
    parser.add_argument("--all", action="store_true", help="Download all six benchmark datasets")
    parser.add_argument("--list", action="store_true", help="Show the catalogue and exit")
    parser.add_argument(
        "--out",
        default="data/datasets",
        help="Destination directory (default: data/datasets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded, and how much disk it needs",
    )
    args = parser.parse_args()

    if args.list or (not args.datasets and not args.all):
        print_catalogue()
        return 0

    keys = list(DATASETS) if args.all else resolve(args.datasets)
    selected = [DATASETS[k] for k in keys]
    needed = sum(d.total_bytes for d in selected)

    out = Path(args.out).expanduser().resolve()
    print(f"\n  Destination : {out}")
    print(f"  Datasets    : {', '.join(keys)}")
    print(f"  Download    : {human(needed)}")

    free = shutil.disk_usage(out.parent if not out.exists() else out).free
    print(f"  Free space  : {human(free)}")
    if free < needed * 1.1:
        print("\n  ✗ Not enough free disk space (want ~10% headroom).", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n  --dry-run, nothing downloaded:\n")
        for ds in selected:
            print(f"  {ds.key}")
            for f in ds.files:
                print(f"    {f.name}  {human(f.size)}")
                print(f"      {f.url}")
        print()
        return 0

    out.mkdir(parents=True, exist_ok=True)

    failures = []
    for ds in selected:
        print(f"\n  {ds.title}  [{ds.modality}]")
        for remote in ds.files:
            if not download(remote, out / remote.name):
                failures.append(remote.name)

    print()
    if failures:
        print(f"  ✗ {len(failures)} file(s) failed: {', '.join(failures)}", file=sys.stderr)
        print("    Re-run the same command; completed files are skipped.", file=sys.stderr)
        return 1

    print(f"  ✅ All files downloaded and verified in {out}\n")
    print("  Next:")
    print(f"    cellforge --dataset-path {out}/<file>.h5ad --task-file examples/<task>.txt\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
