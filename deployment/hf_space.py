from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SPACE_REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    "config/models.yaml",
    "config/training.yaml",
]


@dataclass(frozen=True)
class SpaceBuildStatus:
    required_files_present: bool
    missing_files: list[str]
    readme_has_space_metadata: bool
    has_space_remote: bool

    def as_dict(self) -> dict:
        return asdict(self)


def missing_required_files(root: str | Path = ".") -> list[str]:
    repo_root = Path(root)
    return [path for path in SPACE_REQUIRED_FILES if not (repo_root / path).exists()]


def readme_has_space_metadata(root: str | Path = ".") -> bool:
    readme_path = Path(root) / "README.md"
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    return text.startswith("---\n") and "\nsdk: gradio\n" in text and "\napp_file: app.py\n" in text


def has_space_remote(remote_output: str) -> bool:
    return any(line.startswith("space\t") for line in remote_output.splitlines())


def space_build_status(root: str | Path = ".", remote_output: str = "") -> SpaceBuildStatus:
    missing = missing_required_files(root)
    return SpaceBuildStatus(
        required_files_present=not missing,
        missing_files=missing,
        readme_has_space_metadata=readme_has_space_metadata(root),
        has_space_remote=has_space_remote(remote_output),
    )


def create_space_commands(
    hf_user: str,
    space_name: str = "openbmb-local-ai-workbench",
    branch: str = "main",
) -> list[str]:
    if not hf_user.strip():
        raise ValueError("hf_user is required")
    if not space_name.strip():
        raise ValueError("space_name is required")
    return [
        "huggingface-cli login",
        f"huggingface-cli repo create {space_name} --type space --space-sdk gradio",
        f"git remote add space https://huggingface.co/spaces/{hf_user}/{space_name}",
        f"git push space {branch}",
    ]
