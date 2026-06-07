from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import build_app


def main() -> None:
    demo = build_app()
    try:
        print(type(demo).__name__)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
