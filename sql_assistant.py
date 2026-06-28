"""Deprecated single-file module.

The project has been split into the ``sql_assistant`` package. This file is
kept only so existing imports keep working; please import from the package and
run the app via ``python app.py``.

You can safely delete this file.
"""

from sql_assistant import (  # noqa: F401
    DatabaseConfig,
    SystemConfig,
    ExecutionMode,
    process_question,
    create_ui,
)
