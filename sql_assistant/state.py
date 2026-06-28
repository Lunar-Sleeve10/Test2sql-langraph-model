"""Shared agent state definition."""

from typing import Any, Dict, List, Optional, TypedDict

import plotly.graph_objects as go

from .adapters import DatabaseAdapter
from .config import SystemConfig


class AgentState(TypedDict):
    question: str
    is_clear: bool
    clarification_message: str
    clarification_count: int
    normalized_question: str
    subtasks: List[str]
    complexity: str
    execution_mode: str
    db_adapter: Optional[DatabaseAdapter]
    full_schema_cache: str
    relevant_tables: List[str]
    relevant_columns: Dict[str, List[str]]
    refined_schema_str: str
    analytical_plan: Dict[str, Any]
    sql_query: str
    sql_generation_metadata: Dict[str, Any]
    query_result: Dict[str, Any]
    needs_visualization: bool
    visualization_params: Dict[str, Any]
    chart: Optional[go.Figure]
    error: str
    metadata: Dict[str, Any]
    system_config: Optional[SystemConfig]
    verification_passed: bool
