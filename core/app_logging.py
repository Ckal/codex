from __future__ import annotations

import logging


def configure_app_logging() -> None:
    """Configure compact application logging once."""

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
