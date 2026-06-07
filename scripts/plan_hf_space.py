from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from deployment.hf_space import create_space_commands, space_build_status

    parser = argparse.ArgumentParser(description="Plan Hugging Face Space deployment commands.")
    parser.add_argument("--user", required=True, help="Hugging Face username or org.")
    parser.add_argument("--space", default="openbmb-local-ai-workbench", help="Space repo name.")
    parser.add_argument("--branch", default="main", help="Git branch to push.")
    args = parser.parse_args()

    status = space_build_status()
    print(json.dumps({"status": status.as_dict()}, indent=2))
    print("\nCommands to run manually:")
    for command in create_space_commands(args.user, args.space, args.branch):
        print(command)


if __name__ == "__main__":
    main()
