"""Gradio user interface."""

import os

import gradio as gr

from .adapters import SchemaCache
from .config import SystemConfig
from .orchestrator import process_question


def create_ui() -> gr.Blocks:
    default_db = os.getenv("DB_PATH", "chinook.db")
    default_db_type = os.getenv("DB_TYPE", "sqlite")
    default_api_key = os.getenv("GROQ_API_KEY", "")
    default_model = os.getenv("LLM_MODEL", "groq/llama-3.1-70b-versatile")
    default_sql_model = os.getenv("SQL_MODEL", "groq/llama-3.1-70b-versatile")

    with gr.Blocks(title="SQL Analytics Assistant") as demo:
        gr.Markdown("# AI-Powered SQL Analytics Assistant\n\nNatural language to SQL with automatic visualization.")
        clarification_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. Top 10 customers by revenue | Monthly sales trend | Compare Q1 vs Q2",
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("Generate query", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", scale=1)

                with gr.Accordion("Advanced configuration", open=False):
                    with gr.Row():
                        db_type_input = gr.Dropdown(
                            label="Database type",
                            choices=["sqlite", "postgres", "mysql", "bigquery"],
                            value=default_db_type,
                        )
                        db_path = gr.Textbox(label="Database path", value=default_db)
                    api_key_input = gr.Textbox(label="API key", value=default_api_key, type="password")
                    with gr.Row():
                        model_input = gr.Textbox(label="Main model", value=default_model)
                        sql_model_input = gr.Textbox(label="SQL model", value=default_sql_model)
                    with gr.Row():
                        temperature_input = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.1, step=0.05)
                        max_rows_input = gr.Slider(label="Max rows", minimum=100, maximum=5000, value=1000, step=100)
                    clear_cache_btn = gr.Button("Clear schema cache", size="sm")

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Results"):
                        response_output = gr.Textbox(label="Summary", lines=12, interactive=False)
                        chart_output = gr.Plot(label="Visualization")
                    with gr.Tab("Data"):
                        results_table = gr.Dataframe(label="Query results", interactive=False, wrap=True)

        gr.Markdown(
            "Tips: rankings (top 10, lowest), trends (monthly, over time), "
            "comparisons (vs, year-over-year), distributions (share, breakdown)."
        )

        def process_query(question, db_type, db_path_val, api_key, model_name,
                           sql_model, temperature, max_rows, clarif_count):
            if not api_key.strip():
                return "Please provide an API key in Advanced configuration.", None, None, 0
            if not db_path_val.strip():
                return "Please provide a database path.", None, None, 0
            if not question.strip():
                return "Please enter a question.", None, None, 0

            config = SystemConfig(
                llm_provider="litellm",
                api_key=api_key,
                model_name=model_name,
                sql_model=sql_model,
                temperature=temperature,
                max_clarifications=3,
                max_rows=int(max_rows),
            )
            response, df, chart, new_clarif = process_question(
                question, db_path_val, db_type, config, clarif_count
            )
            return response, chart, df, new_clarif

        submit_btn.click(
            fn=process_query,
            inputs=[question_input, db_type_input, db_path, api_key_input, model_input,
                    sql_model_input, temperature_input, max_rows_input, clarification_state],
            outputs=[response_output, chart_output, results_table, clarification_state],
        )
        clear_btn.click(
            fn=lambda: ("", None, None, 0),
            outputs=[question_input, chart_output, results_table, clarification_state],
        )
        clear_cache_btn.click(
            fn=lambda: (SchemaCache.clear() or "Schema cache cleared."),
            outputs=[response_output],
        )

    return demo
