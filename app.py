"""Application entry point.

Loads environment variables from a local .env file (if present) and launches
the Gradio UI.
"""

import sys
import logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

from sql_assistant.ui import create_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("SQL Assistant starting...")
    try:
        create_ui().launch(show_error=True, share=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Launch failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
