"""
- logging.Logger
"""

import logging
import os


def setup_logging(
    run_id: str,
    level: int = logging.INFO,
    disable: bool = False,
):
    if disable:
        logging.basicConfig(level=logging.CRITICAL + 1)
        return

    filename = f"results/{run_id}.log"
    os.makedirs("results", exist_ok=True)
    format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format=format,
    )
    # logging.getLogger().addHandler(logging.StreamHandler())

    return
