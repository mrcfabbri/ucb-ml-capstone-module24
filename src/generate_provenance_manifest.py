"""Create a public, hash-only provenance manifest for this final deliverable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

from package_submission import package_files

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "provenance_manifest.json"
PRIVATE_CACHE = Path(
    os.environ.get("CAPSTONE_UPSTREAM_CACHE_DIR", ROOT / ".private_cache")
).resolve()
PRIVATE_CHECKPOINT_NAMES = (
    "03_engineered_customer_month.parquet",
    "05b_scored_customer_month.parquet",
    "06_next_state_customer_month.parquet",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def metadata(path: Path, *, public_path: str) -> dict:
    if not path.is_file():
        return {"path": public_path, "available": False}
    return {
        "path": public_path,
        "available": True,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def public_files() -> list[Path]:
    """Return curated public files without the self-referential manifest."""

    return [path for path in package_files() if path.name != OUTPUT.name]


def build_manifest() -> dict:
    files = public_files()
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "package": {
            "path": ".",
            "public_file_count_excluding_manifest": len(files),
            "manifest_scope": "all curated public files except this manifest and the ZIP",
            "freshness_note": (
                "The ZIP is generated from the same unchanged files immediately after this "
                "manifest. validate_artifacts.py verifies every listed hash and archive member."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "uv_project": ".",
            "lockfile": metadata(ROOT / "uv.lock", public_path="uv.lock"),
        },
        "public_submission_files": [
            metadata(path, public_path=str(path.relative_to(ROOT))) for path in files
        ],
        "private_checkpoints_hash_only": [
            metadata(
                PRIVATE_CACHE / name,
                public_path=f".private_cache/{name}",
            )
            for name in PRIVATE_CHECKPOINT_NAMES
        ],
        "privacy_note": (
            "Private checkpoint contents, customer identifiers, commercial values, and row-level "
            "outputs are intentionally omitted. Availability, size, and SHA-256 support an "
            "authorized local rerun without disclosing private contents."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    print(f"Wrote public hash-only provenance manifest: {args.output}")


if __name__ == "__main__":
    main()
