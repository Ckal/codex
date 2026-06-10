from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.hf_space import (  # noqa: E402
    create_space_commands,
    hackathon_space_commands,
    space_build_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan Hugging Face Space deployment")
    parser.add_argument("--user", required=True, help="Hugging Face user or org")
    parser.add_argument("--space", default="openbmb-local-ai-workbench")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    git_executable = shutil.which("git") or "git"
    remote_output = subprocess.run(  # noqa: S603
        [git_executable, "remote", "-v"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    status = space_build_status(".", remote_output)

    print("Space build status:")
    print(status.as_dict())

    print("\nCommands to run manually:")
    for command in create_space_commands(args.user, args.space, args.branch):
        print(f"- {command}")

    print("\nBuild Small Hackathon target Spaces:")
    for name, commands in hackathon_space_commands(args.branch).items():
        print(f"\n{name}:")
        for command in commands:
            print(f"- {command}")


if __name__ == "__main__":
    main()
