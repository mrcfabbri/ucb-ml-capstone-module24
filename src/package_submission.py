"""Build a curated ZIP without private caches or local/editor residue."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "module24_deliverable.zip"
INCLUDED_DIRECTORIES = ("notebooks", "src", "tests", "docs", "figures")
INCLUDED_FILES = (
    ".gitignore",
    "README.md",
    "report.md",
    "evaluation.md",
    "Makefile",
    "pyproject.toml",
    "provenance_manifest.json",
    "uv.lock",
)
EXCLUDED_NAMES = {".DS_Store", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
ALLOWED_DIRECTORY_SUFFIXES = {".ipynb", ".md", ".png", ".py"}


def is_allowed_directory_file(path: Path) -> bool:
    """Allow only the expected public artifact types from recursive directories."""

    return (
        path.suffix in ALLOWED_DIRECTORY_SUFFIXES
        and not any(part in EXCLUDED_NAMES for part in path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and path.name != ".env"
    )


def package_files() -> list[Path]:
    """Return the small, explicit public inventory used for both ZIP creation and validation.

    This allowlist is intentional: a generic folder archive could include private checkpoints,
    cached row-level outputs, virtual environments, or editor residue even when Git ignores them.
    """
    files = [path for name in INCLUDED_FILES if (path := ROOT / name).is_file()]
    directory_files: list[Path] = []
    for directory in INCLUDED_DIRECTORIES:
        directory_files.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and is_allowed_directory_file(path)
        )
    return sorted(files + directory_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in package_files():
            archive.write(path, path.relative_to(ROOT))
    print(f"Wrote curated submission ZIP with {len(package_files())} files: {args.output}")


if __name__ == "__main__":
    main()
