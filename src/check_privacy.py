"""Fail when notebook or archive outputs expose confidential data."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from package_submission import package_files

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
MAIN_NOTEBOOKS = sorted(NOTEBOOKS.glob("*.ipynb"))
EXPECTED_NOTEBOOKS = {
    "05_peer_strategy_baseline.ipynb",
    "06_hurdle_cadence_model.ipynb",
    "07_alert_episodes_and_pilot.ipynb",
    "99_final_findings.ipynb",
}

FORBIDDEN_OUTPUT_TOKENS = {
    "customer_public_id",
    "dominant_companyid",
    "dominant_divisionid",
    "company_division_peer_public_id",
    "net_revenue",
    "order_value",
    "gross_revenue",
    "total_quantity",
    "total_cost",
    "avg_net_price",
}
FORBIDDEN_ARCHIVE_PARTS = {
    ".cache",
    ".env",
    ".private_cache",
    ".venv",
    "__pycache__",
}
FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".csv",
    ".parquet",
    ".pyc",
    ".pyo",
    ".xlsx",
    ".xls",
}
TEXT_SUFFIXES = {".md", ".py", ".ipynb", ".toml", ".json"}
LOCAL_PATH_PATTERNS = (
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"(?:^|[^A-Za-z0-9_])Dropbox" + r"/GitHub/"),
    re.compile(r"file://(?:/Users|[^\s)]+/Users)/"),
)


def output_text(output: dict) -> str:
    parts = []
    if "text" in output:
        value = output["text"]
        parts.append("".join(value) if isinstance(value, list) else str(value))
    for mime_type, value in output.get("data", {}).items():
        if mime_type.startswith("image/"):
            continue
        if isinstance(value, list):
            parts.append("".join(str(item) for item in value))
        else:
            parts.append(str(value))
    if output.get("output_type") == "error":
        parts.append(str(output.get("ename")))
        parts.append(str(output.get("evalue")))
    return "\n".join(parts).lower()


def cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def scan_portable_text(text: str, *, label: str) -> list[str]:
    """Reject machine-local paths from every included textual artifact."""

    return [
        f"{label}: exposes a machine-local path"
        for pattern in LOCAL_PATH_PATTERNS
        if pattern.search(text)
    ]


def scan_notebook(notebook: dict, *, label: str) -> list[str]:
    failures = []
    for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
        failures.extend(
            scan_portable_text(cell_source(cell), label=f"{label}: cell {cell_number} source")
        )
        for output in cell.get("outputs", []):
            text = output_text(output)
            failures.extend(
                scan_portable_text(text, label=f"{label}: cell {cell_number} output")
            )
            for token in FORBIDDEN_OUTPUT_TOKENS:
                if token in text:
                    failures.append(
                        f"{label}: cell {cell_number}: forbidden output token {token}"
                    )
            if output.get("output_type") == "error":
                failures.append(f"{label}: cell {cell_number}: saved error output")
    return failures


def scan_live_notebooks() -> list[str]:
    failures = []
    names = {path.name for path in MAIN_NOTEBOOKS}
    if names != EXPECTED_NOTEBOOKS:
        failures.append(
            f"Expected final notebooks {sorted(EXPECTED_NOTEBOOKS)}; found {sorted(names)}"
        )
    for path in MAIN_NOTEBOOKS:
        failures.extend(scan_notebook(json.loads(path.read_text()), label=path.name))
    return failures


def scan_live_public_files() -> list[str]:
    """Scan all included textual files, not merely the notebook outputs."""

    failures = []
    for path in package_files():
        if path.suffix in TEXT_SUFFIXES and path.suffix != ".ipynb":
            failures.extend(
                scan_portable_text(path.read_text(), label=str(path.relative_to(ROOT)))
            )
    return failures


def scan_archive(path: Path) -> list[str]:
    """Apply the same privacy checks to the final ZIP, not just the source directory."""
    failures = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            failures.append(f"{path}: duplicate archive member names")
        for name in names:
            member = Path(name)
            if any(part in FORBIDDEN_ARCHIVE_PARTS for part in member.parts):
                failures.append(f"{path}: forbidden archive path {name}")
            if member.suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES:
                failures.append(f"{path}: forbidden archive type {name}")
            if member.suffix == ".ipynb":
                notebook = json.loads(archive.read(name))
                failures.extend(scan_notebook(notebook, label=f"{path.name}:{name}"))
            elif member.suffix in TEXT_SUFFIXES:
                failures.extend(
                    scan_portable_text(
                        archive.read(name).decode("utf-8"), label=f"{path.name}:{name}"
                    )
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    failures = scan_live_notebooks() + scan_live_public_files()
    if args.archive:
        failures.extend(scan_archive(args.archive))
    if failures:
        raise SystemExit("\n".join(failures))
    archive_note = f" and archive {args.archive.name}" if args.archive else ""
    print(
        "OK: scanned four final notebooks"
        f"{archive_note}; no saved errors, identifiers, absolute commercial fields, "
        "private cache paths, or private data file types were exposed."
    )


if __name__ == "__main__":
    main()
