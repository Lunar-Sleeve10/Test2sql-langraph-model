"""End-to-end orchestration: build adapter, run the workflow, format output."""

import logging
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from .adapters import DatabaseAdapter, SchemaCache, build_adapter
from .config import DatabaseConfig, ExecutionMode, SystemConfig
from .state import AgentState
from .workflow import create_workflow

logger = logging.getLogger(__name__)


def process_question(
    question: str,
    db_path: str,
    db_type: str,
    system_config: SystemConfig,
    clarification_count: int = 0,
) -> Tuple[str, Optional[pd.DataFrame], Optional[go.Figure], int]:
    if not question.strip():
        return "Please enter a question.", None, None, 0

    adapter: Optional[DatabaseAdapter] = None
    try:
        db_config = DatabaseConfig(db_type, db_path, max_rows=system_config.max_rows)
        adapter = build_adapter(db_config)
        full_schema = SchemaCache.get_schema(db_path, adapter)
    except Exception as exc:  # noqa: BLE001
        return f"Database error: {exc}", None, None, 0

    try:
        initial: AgentState = {
            "question": question,
            "is_clear": False,
            "clarification_message": "",
            "clarification_count": clarification_count,
            "normalized_question": "",
            "subtasks": [],
            "complexity": "medium",
            "execution_mode": ExecutionMode.LIGHT_PLAN.value,
            "db_adapter": adapter,
            "full_schema_cache": full_schema,
            "relevant_tables": [],
            "relevant_columns": {},
            "refined_schema_str": "",
            "analytical_plan": {},
            "sql_query": "",
            "sql_generation_metadata": {},
            "query_result": {"data": [], "columns": []},
            "needs_visualization": False,
            "visualization_params": {},
            "chart": None,
            "error": "",
            "metadata": {},
            "system_config": system_config,
            "verification_passed": False,
        }

        final = create_workflow().invoke(initial)

        if final.get("error"):
            response = f"Error: {final['error']}"
            if final.get("sql_query"):
                response += f"\n\nGenerated SQL:\n```sql\n{final['sql_query']}\n```"
            return response, None, None, 0

        if not final.get("is_clear"):
            return (
                final.get("clarification_message", "Unclear request."),
                None,
                None,
                final.get("clarification_count", 0),
            )

        rows = final["query_result"]["data"]
        cols = final["query_result"]["columns"]
        if not rows:
            return f"No results found.\n\nSQL:\n```sql\n{final['sql_query']}\n```", None, None, 0

        df = pd.DataFrame(rows, columns=cols)
        subtasks = "\n".join(f"{i+1}. {t}" for i, t in enumerate(final.get("subtasks", [])))
        meta = final.get("sql_generation_metadata", {})
        total_time = sum(v for v in final["metadata"].values() if isinstance(v, (int, float)))

        response = (
            "Query completed.\n\n"
            f"Question: {final['normalized_question']}\n"
            f"Complexity: {final['complexity']}\n"
            f"Pattern: {final.get('metadata', {}).get('query_pattern', 'unknown')}\n\n"
            f"Subtasks:\n{subtasks}\n\n"
            f"Tables: {', '.join(final['relevant_tables'])}\n"
            f"SQL model: {meta.get('model', 'N/A')} (attempt {meta.get('attempt_number', 1)})\n\n"
            f"SQL:\n```sql\n{final['sql_query']}\n```\n\n"
            f"Results: {len(df)} rows x {len(df.columns)} columns\n"
            f"Total time: {total_time:.2f}s"
        )
        return response, df, final.get("chart"), 0

    except Exception as exc:  # noqa: BLE001
        logger.error("Process error: %s", exc, exc_info=True)
        return f"System error: {exc}", None, None, 0
    finally:
        if adapter:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass
