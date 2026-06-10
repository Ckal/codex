from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SPACE_REQUIRED_FILES = [
    "app.py",
    "plant_space_app.py",
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


def has_space_remote(remote_output: str, remote_name: str = "space") -> bool:
    return any(line.startswith(f"{remote_name}\t") for line in remote_output.splitlines())


def space_build_status(root: str | Path = ".", remote_output: str = "") -> SpaceBuildStatus:
    missing = missing_required_files(root)
    return SpaceBuildStatus(
        required_files_present=not missing,
        missing_files=missing,
        readme_has_space_metadata=readme_has_space_metadata(root),
        has_space_remote=(
            has_space_remote(remote_output, "space")
            or has_space_remote(remote_output, "space-workbench")
            or has_space_remote(remote_output, "space-plant")
        ),
    )


def create_space_commands(
    hf_user: str,
    space_name: str = "openbmb-local-ai-workbench",
    branch: str = "main",
    remote_name: str = "space",
    app_file: str = "app.py",
) -> list[str]:
    if not hf_user.strip():
        raise ValueError("hf_user is required")
    if not space_name.strip():
        raise ValueError("space_name is required")
    if not remote_name.strip():
        raise ValueError("remote_name is required")
    if not app_file.strip():
        raise ValueError("app_file is required")
    return [
        "hf auth login",
        f"hf repo create {hf_user}/{space_name} --type space --space-sdk gradio",
        f"git remote add {remote_name} https://huggingface.co/spaces/{hf_user}/{space_name}",
        f"git push {remote_name} {branch}",
        f"Space metadata app_file must be {app_file} for this deployment.",
    ]


def hackathon_space_commands(branch: str = "main") -> dict[str, list[str]]:
    return {
        "workbench": create_space_commands(
            "build-small-hackathon",
            "workbench",
            branch,
            remote_name="space-workbench",
            app_file="app.py",
        ),
        "plant": create_space_commands(
            "build-small-hackathon",
            "plant_identification_tool",
            branch,
            remote_name="space-plant",
            app_file="plant_space_app.py",
        ),
    }
