"""Validate notebooks, figures, links, provenance, and the curated archive."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import nbformat
from PIL import Image

from package_submission import package_files

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance_manifest.json"
EXPECTED_NOTEBOOK_IMAGES = {
    "05_peer_strategy_baseline.ipynb": 4,
    "06_hurdle_cadence_model.ipynb": 8,
    "07_alert_episodes_and_pilot.ipynb": 3,
    "99_final_findings.ipynb": 2,
}
MARKDOWN_FILES = (
    ROOT / "README.md",
    ROOT / "report.md",
    ROOT / "evaluation.md",
    ROOT / "docs" / "business_value_validation.md",
)
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
LOCAL_PATH_TOKENS = ("/Users/", "Dropbox/GitHub", "file://")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_path(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def validate_png(payload: bytes, *, label: str) -> None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
            if image.format != "PNG":
                raise ValueError(f"expected PNG, found {image.format}")
    except Exception as exc:
        raise ValueError(f"{label}: image cannot be decoded and verified: {exc}") from exc


def joined_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def joined_output_text(output: dict) -> str:
    parts = []
    if "text" in output:
        value = output["text"]
        parts.append("".join(value) if isinstance(value, list) else str(value))
    for mime_type, value in output.get("data", {}).items():
        if mime_type.startswith("image/"):
            continue
        parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def validate_notebook(payload: bytes, *, label: str, expected_images: int) -> int:
    """Check that a submitted notebook is readable, executed, and visually complete.

    The checks deliberately inspect saved evidence rather than merely importing code: a reviewer
    should receive sequentially executed notebooks, no hidden errors, and the promised plots.
    """
    notebook = nbformat.reads(payload.decode("utf-8"), as_version=4)
    nbformat.validate(notebook)

    headings = [
        joined_source(cell)
        for cell in notebook.cells
        if cell.cell_type == "markdown" and joined_source(cell).lstrip().startswith("#")
    ]
    if len(headings) < 4:
        raise ValueError(f"{label}: expected at least four formatted headings")

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    counts = [cell.execution_count for cell in code_cells]
    if counts != list(range(1, len(code_cells) + 1)):
        raise ValueError(f"{label}: execution counts are not sequential: {counts}")

    image_count = 0
    for cell_number, cell in enumerate(notebook.cells, start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise ValueError(f"{label}: cell {cell_number} contains a saved error")
            text = joined_output_text(output)
            if output.get("output_type") == "stream" and len(text.splitlines()) > 80:
                raise ValueError(f"{label}: cell {cell_number} has excessive stream output")
            if any(token.lower() in text.lower() for token in LOCAL_PATH_TOKENS):
                raise ValueError(f"{label}: cell {cell_number} exposes a local path")
            image_value = output.get("data", {}).get("image/png")
            if image_value is None:
                continue
            image_count += 1
            encoded = "".join(image_value) if isinstance(image_value, list) else image_value
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError(
                    f"{label}: cell {cell_number} contains invalid base64 image data: {exc}"
                ) from exc
            validate_png(decoded, label=f"{label}: cell {cell_number}")

    if image_count != expected_images:
        raise ValueError(
            f"{label}: expected {expected_images} rendered plots; found {image_count}"
        )
    return image_count


def validate_live_artifacts() -> int:
    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    names = {path.name for path in notebook_paths}
    if names != set(EXPECTED_NOTEBOOK_IMAGES):
        raise ValueError(
            f"Unexpected notebook set: expected {sorted(EXPECTED_NOTEBOOK_IMAGES)}, "
            f"found {sorted(names)}"
        )

    image_count = sum(
        validate_notebook(
            path.read_bytes(),
            label=path.name,
            expected_images=EXPECTED_NOTEBOOK_IMAGES[path.name],
        )
        for path in notebook_paths
    )

    for figure in sorted((ROOT / "figures").glob("*.png")):
        validate_png(figure.read_bytes(), label=str(figure.relative_to(ROOT)))

    for markdown_path in MARKDOWN_FILES:
        text = markdown_path.read_text()
        for target in LINK_PATTERN.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if Path(target).is_absolute():
                raise ValueError(f"{markdown_path.name}: absolute link is not portable: {target}")
            local_target = (markdown_path.parent / target.split("#", 1)[0]).resolve()
            if not local_target.exists():
                raise ValueError(f"{markdown_path.name}: broken local link: {target}")
    return image_count


def validate_manifest(manifest: dict) -> None:
    expected_files = {
        str(path.relative_to(ROOT)): path
        for path in package_files()
        if path.name != MANIFEST.name
    }
    records = {record["path"]: record for record in manifest["public_submission_files"]}
    if set(records) != set(expected_files):
        missing = sorted(set(expected_files) - set(records))
        extra = sorted(set(records) - set(expected_files))
        raise ValueError(f"Manifest inventory mismatch; missing={missing}, extra={extra}")
    for relative_path, path in expected_files.items():
        record = records[relative_path]
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"Manifest size is stale for {relative_path}")
        if record.get("sha256") != digest_path(path):
            raise ValueError(f"Manifest hash is stale for {relative_path}")
    if manifest["package"]["public_file_count_excluding_manifest"] != len(expected_files):
        raise ValueError("Manifest public file count is stale")


def validate_archive(path: Path, manifest: dict | None) -> None:
    expected = {
        str(file.relative_to(ROOT)): file.read_bytes() for file in package_files()
    }
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member names")
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise ValueError(f"Archive inventory mismatch; missing={missing}, extra={extra}")
        for name, live_payload in expected.items():
            archived_payload = archive.read(name)
            if archived_payload != live_payload:
                raise ValueError(f"Archive member differs from canonical file: {name}")
            if name.startswith("notebooks/"):
                notebook_name = Path(name).name
                validate_notebook(
                    archived_payload,
                    label=f"{path.name}:{name}",
                    expected_images=EXPECTED_NOTEBOOK_IMAGES[notebook_name],
                )
            elif name.endswith(".png"):
                validate_png(archived_payload, label=f"{path.name}:{name}")

        archived_manifest = json.loads(archive.read("provenance_manifest.json"))
        if manifest is not None and archived_manifest != manifest:
            raise ValueError("Archive manifest differs from canonical manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    image_count = validate_live_artifacts()
    manifest = None
    if args.manifest or args.archive:
        manifest = json.loads(MANIFEST.read_text())
        validate_manifest(manifest)
    if args.archive:
        validate_archive(args.archive, manifest)

    extras = []
    if args.manifest:
        extras.append("manifest hashes")
    if args.archive:
        extras.append("archive inventory and bytes")
    suffix = f"; verified {', '.join(extras)}" if extras else ""
    print(
        f"OK: four notebooks, sequential execution, {image_count} embedded plots, "
        f"external figures, and local links are valid{suffix}."
    )


if __name__ == "__main__":
    main()
