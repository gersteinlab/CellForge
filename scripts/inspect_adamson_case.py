#!/usr/bin/env python3
"""Inspect an Adamson scPerturb h5ad without loading its matrix into memory."""

from pathlib import Path
import argparse
import hashlib
import json

import h5py


EXPECTED_MD5 = "232f7e3756d41602bbe434b50662a76f"


def _keys(group):
    return sorted(str(key) for key in group.keys())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    path = args.dataset.expanduser().resolve()

    digest = hashlib.md5(path.read_bytes()).hexdigest()
    with h5py.File(path, "r") as handle:
        obs = handle.get("obs")
        var = handle.get("var")
        matrix = handle.get("X")
        summary = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "md5": digest,
            "md5_matches_zenodo": digest == EXPECTED_MD5,
            "root_keys": _keys(handle),
            "obs_columns": _keys(obs) if obs is not None else [],
            "var_columns": _keys(var) if var is not None else [],
            "matrix_encoding": matrix.attrs.get("encoding-type", "") if matrix is not None else "",
            "matrix_shape": list(matrix.attrs.get("shape", [])) if matrix is not None else [],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0 if digest == EXPECTED_MD5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
