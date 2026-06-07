from __future__ import annotations

from pathlib import Path


def copy_text_file_or_empty(source_path: str | Path, output_path: str | Path) -> Path:
    """Copy a text file to an export path, or create an empty export."""

    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        output.write_text("", encoding="utf-8")
    return output
