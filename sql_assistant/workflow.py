"""LangGraph workflow assembly and routing."""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    intent_analyzer_node,
    normalizer_reasoning_node,
    table_selector_node,
    verification_node,
    refiner_node,
    planner_node,
    sql_generator_node,
    execute_query_node,
    visualization_decision_node,
    visualization_generator_node,
)


def route_after_intent(state: AgentState) -> str:
    if state.get("error"):
        return END
    return "normalizer" if state.get("is_clear") else END


def route_after_table_select(state: AgentState) -> str:
    return END if state.get("error") else "verification"


def route_after_verification(state: AgentState) -> str:
    return "refiner" if state.get("verification_passed") else END


def route_after_execute(state: AgentState) -> str:
    return END if state.get("error") else "viz_decision"


def route_after_viz_decision(state: AgentState) -> str:
    return "viz_generate" if state.get("needs_visualization") else END


def create_workflow():
    wf = StateGraph(AgentState)
    wf.add_node("intent", intent_analyzer_node)
    wf.add_node("normalizer", normalizer_reasoning_node)
    wf.add_node("table_select", table_selector_node)
    wf.add_node("verification", verification_node)
    wf.add_node("refiner", refiner_node)
    wf.add_node("planner", planner_node)
    wf.add_node("sql_gen", sql_generator_node)
    wf.add_node("execute", execute_query_node)
    wf.add_node("viz_decision", visualization_decision_node)
    wf.add_node("viz_generate", visualization_generator_node)

    wf.set_entry_point("intent")
    wf.add_conditional_edges("intent", route_after_intent, {"normalizer": "normalizer", END: END})
    wf.add_edge("normalizer", "table_select")
    wf.add_conditional_edges("table_select", route_after_table_select, {"verification": "verification", END: END})
    wf.add_conditional_edges("verification", route_after_verification, {"refiner": "refiner", END: END})
    wf.add_edge("refiner", "planner")
    wf.add_edge("planner", "sql_gen")
    wf.add_edge("sql_gen", "execute")
    wf.add_conditional_edges("execute", route_after_execute, {"viz_decision": "viz_decision", END: END})
    wf.add_conditional_edges("viz_decision", route_after_viz_decision, {"viz_generate": "viz_generate", END: END})
    wf.add_edge("viz_generate", END)
    return wf.compile()
