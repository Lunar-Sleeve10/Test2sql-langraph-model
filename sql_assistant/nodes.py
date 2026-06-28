"""LangGraph node implementations for the SQL assistant pipeline."""

import re
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .adapters import DatabaseAdapter
from .config import ExecutionMode
from .llm import call_llm
from .parsing import LLMResponseParser, SQLValidator
from .state import AgentState

logger = logging.getLogger(__name__)


def _ensure_metadata(state: AgentState) -> None:
    if state.get("metadata") is None:
        state["metadata"] = {}


def intent_analyzer_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    question = state["question"]
    config = state["system_config"]
    count = state.get("clarification_count", 0)

    if count >= config.max_clarifications:
        state["is_clear"] = False
        state["clarification_message"] = f"Maximum clarifications reached ({config.max_clarifications})."
        return state

    prompt = f"""Decide whether this is a clear analytical SQL request.

Question: "{question}"

A request is CLEAR when it names a metric and a dimension (e.g. ranking,
share/percentage, comparison, aggregation, distribution, trend, or filter).
It is UNCLEAR when it is vague, a greeting, or missing a metric/dimension.

Examples CLEAR: "top 10 customers by revenue", "market share by category",
"compare Q1 vs Q2", "total sales by region".
Examples UNCLEAR: "show data", "sales", "hello".

Return JSON only:
{{"is_clear": true/false, "clarification_message": "what to ask, or empty",
"detected_pattern": "ranking|share|comparison|aggregation|distribution|trend|filter|unclear"}}"""

    try:
        result = LLMResponseParser.extract_json(call_llm(config, prompt))
        if result:
            state["is_clear"] = bool(result.get("is_clear", False))
            state["clarification_message"] = result.get("clarification_message", "")
            state["metadata"]["query_pattern"] = result.get("detected_pattern", "unknown")
        else:
            state["is_clear"] = False
            state["clarification_message"] = "Please rephrase with a clear metric and dimension."
            state["metadata"]["query_pattern"] = "parse_error"
    except Exception as exc:  # noqa: BLE001
        logger.error("Intent analysis failed: %s", exc)
        state["is_clear"] = False
        state["clarification_message"] = "Please provide more detail about what to analyze."
        state["metadata"]["query_pattern"] = "error"

    if not state["is_clear"]:
        state["clarification_count"] = count + 1
        if not state["clarification_message"]:
            state["clarification_message"] = "Please specify the metric and dimension to analyze."

    state["metadata"]["intent_time"] = (datetime.now() - start).total_seconds()
    logger.info("Intent clear=%s pattern=%s", state["is_clear"], state["metadata"]["query_pattern"])
    return state


def normalizer_reasoning_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    question = state["question"]
    config = state["system_config"]

    prompt = f"""Rephrase the question clearly and break it into logical steps.

Question: {question}

Return JSON only:
{{"normalized_question": "clear rephrasing",
"subtasks": ["logical step 1", "logical step 2"],
"complexity": "simple|medium|complex"}}

Subtasks are logical steps, not SQL subqueries.
Complexity: simple = lookup/filter/count; medium = aggregation/grouping/joins;
complex = multiple operations or window functions."""

    try:
        result = LLMResponseParser.extract_json(call_llm(config, prompt)) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Normalization failed: %s", exc)
        result = {}

    state["normalized_question"] = result.get("normalized_question", question)
    state["subtasks"] = result.get("subtasks") or [question]
    state["complexity"] = result.get("complexity", "medium")

    state["execution_mode"] = {
        "simple": ExecutionMode.DIRECT_SQL.value,
        "complex": ExecutionMode.FULL_PLAN.value,
    }.get(state["complexity"], ExecutionMode.LIGHT_PLAN.value)

    state["metadata"]["normalize_time"] = (datetime.now() - start).total_seconds()
    logger.info("Subtasks: %s", state["subtasks"])
    return state


def _semantic_table_fallback(state: AgentState, adapter: DatabaseAdapter, all_tables: List[str]) -> None:
    """Keyword-overlap scoring fallback when LLM selection fails."""
    question = state["normalized_question"]
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()))
    scores: Dict[str, int] = {}

    for table in all_tables:
        score = 0
        tl = table.lower()
        for w in words:
            if w == tl or w in tl or tl in w:
                score += 10
        try:
            for col in adapter.get_columns(table):
                cl = col.lower()
                for w in words:
                    if w == cl or w in cl or cl in w:
                        score += 2
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read columns for %s: %s", table, exc)
        if score:
            scores[table] = score

    if scores:
        selected = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]]
    else:
        selected = all_tables[:3]

    state["relevant_tables"] = selected
    state["relevant_columns"] = {}
    for table in selected:
        try:
            state["relevant_columns"][table] = adapter.get_columns(table)
        except Exception as exc:  # noqa: BLE001
            logger.error("Fallback column fetch failed for %s: %s", table, exc)
            state["relevant_columns"][table] = []


def table_selector_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    config = state["system_config"]
    adapter = state["db_adapter"]
    all_tables = adapter.get_tables()
    subtasks_str = "\n".join(f"- {t}" for t in state["subtasks"])

    dd_context = ""
    if config.data_dictionary:
        dd_context = f"\n\nData dictionary:\n{json.dumps(config.data_dictionary, indent=2)}"

    prompt = f"""Select the tables needed to answer the question.

Question: {state['normalized_question']}
Subtasks:
{subtasks_str}

Available tables:
{chr(10).join('- ' + t for t in all_tables)}

Schema:
{state['full_schema_cache']}{dd_context}

Steps: extract key entities, match them to table names, then to columns.
Return JSON only:
{{"relevant_tables": ["table1", "table2"], "reasoning": "brief"}}

Only use table names from the available list. Never return an empty list."""

    try:
        result = LLMResponseParser.extract_json(call_llm(config, prompt))
        selected = (result or {}).get("relevant_tables") or []
        if not selected:
            raise ValueError("LLM returned no tables")

        state["relevant_tables"] = []
        state["relevant_columns"] = {}
        for table in selected:
            if table in all_tables:
                state["relevant_tables"].append(table)
                state["relevant_columns"][table] = adapter.get_columns(table)
            else:
                logger.warning("Table %s not in database, ignoring", table)

        if not state["relevant_tables"]:
            raise ValueError("None of the selected tables exist")

        col_prompt = f"""Select the columns needed for this query.

Question: {state['normalized_question']}
Subtasks: {subtasks_str}

Available columns:
{json.dumps(state['relevant_columns'], indent=2)}

Return JSON only:
{{"relevant_columns": {{"table1": ["col1", "col2"]}}}}
Use exact column names. Do not invent columns."""

        col_result = LLMResponseParser.extract_json(call_llm(config, col_prompt))
        if col_result and col_result.get("relevant_columns"):
            verified: Dict[str, List[str]] = {}
            for table, cols in col_result["relevant_columns"].items():
                if table in state["relevant_columns"]:
                    actual = state["relevant_columns"][table]
                    keep = [c for c in cols if c in actual]
                    verified[table] = keep if keep else actual
            if verified:
                state["relevant_columns"] = verified

    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM table selection failed (%s); using semantic fallback", exc)
        _semantic_table_fallback(state, adapter, all_tables)

    if not state["relevant_tables"]:
        state["error"] = "Could not select any valid tables from the database."
        logger.error("No tables selected.")
    else:
        logger.info("Selected tables: %s", state["relevant_tables"])

    state["metadata"]["table_select_time"] = (datetime.now() - start).total_seconds()
    return state


def verification_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    adapter = state["db_adapter"]
    actual_tables = adapter.get_tables()

    valid_tables: List[str] = []
    valid_columns: Dict[str, List[str]] = {}

    for table in state["relevant_tables"]:
        if table not in actual_tables:
            logger.warning("Table %s does not exist", table)
            continue
        valid_tables.append(table)
        actual_cols = adapter.get_columns(table)
        selected = state["relevant_columns"].get(table, [])
        keep = [c for c in selected if c in actual_cols]
        valid_columns[table] = keep if keep else actual_cols

    if not valid_tables:
        state["verification_passed"] = False
        state["error"] = "No valid tables found in the database."
    else:
        state["verification_passed"] = True
        state["relevant_tables"] = valid_tables
        state["relevant_columns"] = valid_columns

    state["metadata"]["verification_time"] = (datetime.now() - start).total_seconds()
    return state


def refiner_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    adapter = state["db_adapter"]

    for table in state["relevant_tables"]:
        try:
            db_cols = adapter.get_columns(table)
            current = state["relevant_columns"].get(table, [])
            verified = [c for c in current if c in db_cols]
            state["relevant_columns"][table] = verified if verified else db_cols
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read columns for %s: %s", table, exc)

    base_schema = adapter.get_schema_info(state["relevant_tables"])
    lines = [
        "DATABASE SCHEMA",
        base_schema,
        "",
        "EXACT COLUMN NAMES (use these verbatim, case-sensitive):",
    ]
    for table in state["relevant_tables"]:
        cols = state["relevant_columns"].get(table, [])
        lines.append(f"Table {table} ({len(cols)} columns): {', '.join(cols)}")

    state["refined_schema_str"] = "\n".join(lines)
    state["metadata"]["refine_time"] = (datetime.now() - start).total_seconds()
    return state


def planner_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)

    if state["execution_mode"] == ExecutionMode.DIRECT_SQL.value:
        state["analytical_plan"] = {"description": "Direct query", "subtasks": state["subtasks"]}
        state["metadata"]["plan_time"] = (datetime.now() - start).total_seconds()
        return state

    config = state["system_config"]
    subtasks_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(state["subtasks"]))
    prompt = f"""Create a concise query plan.

Schema:
{state['refined_schema_str']}

Question: {state['normalized_question']}
Subtasks:
{subtasks_str}

Return JSON only:
{{"description": "approach", "operations": ["op1", "op2"],
"join_strategy": "how tables connect", "aggregations": ["what to aggregate"]}}"""

    try:
        result = LLMResponseParser.extract_json(call_llm(config, prompt)) or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Planning failed: %s", exc)
        result = {}

    result["subtasks"] = state["subtasks"]
    result.setdefault("description", "Standard aggregation")
    state["analytical_plan"] = result
    state["metadata"]["plan_time"] = (datetime.now() - start).total_seconds()
    return state


def sql_generator_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    config = state["system_config"]
    adapter = state["db_adapter"]
    available = adapter.get_tables()

    valid_tables = [t for t in state["relevant_tables"] if t in available]
    if not valid_tables:
        state["error"] = "None of the selected tables exist in the database."
        state["sql_query"] = ""
        state["metadata"]["sql_gen_time"] = (datetime.now() - start).total_seconds()
        return state

    columns_info = "\n".join(
        f"{t}: {', '.join(state['relevant_columns'].get(t, []))}" for t in valid_tables
    )
    prompt = f"""Generate a single SQLite SELECT query.

Question: {state['normalized_question']}

Tables and columns:
{columns_info}

Steps:
{chr(10).join('- ' + t for t in state['subtasks'][:3])}

Rules:
1. Use only the tables and columns listed above, with exact names.
2. Valid SQLite syntax only; a single SELECT statement.
3. Use GROUP BY when aggregating and ORDER BY for rankings.
4. Use window functions where they are more efficient.
5. Add LIMIT for top-N requests.

Return the query inside a ```sql``` block."""

    sql_model = config.sql_model or config.model_name
    max_retries = 2

    for attempt in range(max_retries):
        try:
            logger.info("Generating SQL with %s (attempt %d/%d)", sql_model, attempt + 1, max_retries)
            response = call_llm(config, prompt, sql_model)
            sql = LLMResponseParser.extract_sql(response or "")

            if not sql:
                if attempt < max_retries - 1:
                    time.sleep(6)
                    continue
                state["error"] = "Could not extract SQL from the model response."
                state["sql_query"] = ""
                break

            ok, msg = SQLValidator.validate(sql)
            if not ok:
                logger.warning("SQL rejected: %s", msg)
                if attempt < max_retries - 1:
                    continue
                state["error"] = f"Generated SQL was unsafe: {msg}"
                state["sql_query"] = ""
                break

            sql = SQLValidator.enforce_limit(sql, config.max_rows)
            state["sql_query"] = sql
            state["sql_generation_metadata"] = {"model": sql_model, "attempt_number": attempt + 1}
            logger.info("SQL generated successfully.")
            break

        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if "rate_limit" in err.lower() and attempt < max_retries - 1:
                m = re.search(r"try again in ([\d.]+)\s*(ms|s)", err)
                wait = 6.0
                if m:
                    val = float(m.group(1))
                    wait = (val / 1000 if m.group(2) == "ms" else val) + 1.0
                logger.warning("Rate limited; waiting %.1fs", wait)
                time.sleep(wait)
                continue
            if attempt < max_retries - 1:
                logger.warning("SQL generation error: %s; retrying", exc)
                time.sleep(6)
                continue
            state["error"] = f"SQL generation failed: {err[:150]}"
            state["sql_query"] = ""

    state["metadata"]["sql_gen_time"] = (datetime.now() - start).total_seconds()
    return state


def execute_query_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    adapter = state["db_adapter"]
    config = state["system_config"]

    if state.get("error") or not state.get("sql_query", "").strip():
        if not state.get("error"):
            state["error"] = "No SQL query was generated."
        state["query_result"] = {"data": [], "columns": []}
        state["metadata"]["exec_time"] = (datetime.now() - start).total_seconds()
        return state

    try:
        rows, cols = adapter.execute_query(state["sql_query"], config.max_rows)
        state["query_result"] = {"data": rows, "columns": cols}
        logger.info("Retrieved %d rows x %d columns", len(rows), len(cols))
    except Exception as exc:  # noqa: BLE001
        logger.error("Query execution failed: %s", exc)
        state["error"] = f"SQL execution error: {exc}"
        state["query_result"] = {"data": [], "columns": []}

    state["metadata"]["exec_time"] = (datetime.now() - start).total_seconds()
    return state


def visualization_decision_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)
    config = state["system_config"]
    data = state["query_result"]["data"]
    columns = state["query_result"]["columns"]

    if not data:
        state["needs_visualization"] = False
        state["visualization_params"] = {}
        return state

    df = pd.DataFrame(data, columns=columns)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(df) == 1 and len(columns) == 1:
        state["needs_visualization"] = False
        state["visualization_params"] = {}
        return state

    stats = {
        c: {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "mean": float(df[c].mean()),
            "unique": int(df[c].nunique()),
        }
        for c in numeric_cols[:5]
    }

    prompt = f"""Choose the best visualization for this result.

Question: {state['normalized_question']}
Rows: {len(df)}
Columns: {columns}
Numeric: {numeric_cols}
Categorical: {categorical_cols}
Stats: {json.dumps(stats)}
Sample: {json.dumps(df.head(10).to_dict('records'), default=str)[:1200]}

Guidance: pie for part-to-whole shares (3-8 categories); bar for category
comparison or ranking; horizontal_bar for long labels or many categories;
line for time series; scatter for correlation; stacked_bar/grouped_bar for
subcategory breakdowns. Use none for a single value or a large unstructured dump.

Return JSON only:
{{"needs_visualization": true/false,
"chart_type": "pie|bar|horizontal_bar|line|scatter|stacked_bar|grouped_bar|none",
"x_column": "exact column", "y_column": "exact column",
"color_column": "optional column or null", "orientation": "v|h",
"title": "title", "x_label": "label", "y_label": "label",
"show_values": true/false, "reasoning": "brief"}}"""

    try:
        result = LLMResponseParser.extract_json(call_llm(config, prompt))
        if result and result.get("needs_visualization"):
            available = set(columns)
            for key in ("x_column", "y_column", "color_column"):
                col = result.get(key)
                if col and col not in available:
                    match = next((c for c in available if c.lower() == col.lower()), None)
                    if match:
                        result[key] = match
                    elif key in ("x_column", "y_column"):
                        state["needs_visualization"] = False
                        state["visualization_params"] = {}
                        state["metadata"]["viz_decision_time"] = (datetime.now() - start).total_seconds()
                        return state
            state["needs_visualization"] = True
            state["visualization_params"] = result
            logger.info("Visualization: %s", result.get("chart_type"))
        else:
            state["needs_visualization"] = False
            state["visualization_params"] = {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Visualization decision failed: %s", exc)
        state["needs_visualization"] = False
        state["visualization_params"] = {}

    state["metadata"]["viz_decision_time"] = (datetime.now() - start).total_seconds()
    return state


def visualization_generator_node(state: AgentState) -> AgentState:
    start = datetime.now()
    _ensure_metadata(state)

    if not state.get("needs_visualization"):
        state["chart"] = None
        return state

    data = state["query_result"]["data"]
    columns = state["query_result"]["columns"]
    if not data:
        state["chart"] = None
        return state

    df = pd.DataFrame(data, columns=columns)
    p = state["visualization_params"]
    chart_type = p.get("chart_type", "bar")
    x_col = p.get("x_column") or columns[0]
    y_col = p.get("y_column") or (columns[1] if len(columns) > 1 else columns[0])
    color_col = p.get("color_column")
    if color_col not in df.columns:
        color_col = None
    title = p.get("title", "Analysis Results")
    x_label = p.get("x_label", x_col.replace("_", " ").title())
    y_label = p.get("y_label", y_col.replace("_", " ").title())
    show_values = p.get("show_values", True)
    orientation = p.get("orientation", "v")
    palette = px.colors.qualitative.Set2

    try:
        fig: Optional[go.Figure] = None

        if chart_type == "pie":
            fig = go.Figure(
                go.Pie(
                    labels=df[x_col],
                    values=df[y_col],
                    hole=0.3,
                    textinfo="label+percent",
                    marker=dict(colors=palette, line=dict(color="white", width=2)),
                )
            )

        elif chart_type in ("bar", "horizontal_bar"):
            horizontal = chart_type == "horizontal_bar" or orientation == "h"
            fig = go.Figure(
                go.Bar(
                    x=df[y_col] if horizontal else df[x_col],
                    y=df[x_col] if horizontal else df[y_col],
                    orientation="h" if horizontal else "v",
                    marker=dict(color=df[y_col], colorscale="Viridis"),
                    text=df[y_col] if show_values else None,
                    textposition="outside",
                    texttemplate="%{text:.2f}" if show_values else None,
                )
            )
            fig.update_layout(
                xaxis_title=y_label if horizontal else x_label,
                yaxis_title=x_label if horizontal else y_label,
            )

        elif chart_type == "line":
            fig = go.Figure()
            if color_col:
                for cat in df[color_col].unique():
                    sub = df[df[color_col] == cat]
                    fig.add_trace(
                        go.Scatter(x=sub[x_col], y=sub[y_col], mode="lines+markers", name=str(cat))
                    )
            else:
                fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers"))
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, hovermode="x unified")

        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

        elif chart_type in ("stacked_bar", "grouped_bar"):
            barmode = "stack" if chart_type == "stacked_bar" else "group"
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode=barmode, title=title)
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

        if fig is not None:
            fig.update_layout(
                title=dict(text=title, font=dict(size=18)),
                template="plotly_white",
                height=550,
                margin=dict(l=70, r=70, t=80, b=70),
            )
        state["chart"] = fig
    except Exception as exc:  # noqa: BLE001
        logger.error("Visualization generation failed: %s", exc)
        state["chart"] = None

    state["metadata"]["viz_gen_time"] = (datetime.now() - start).total_seconds()
    return state
