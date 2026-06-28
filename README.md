# AI-Powered SQL Analytics Assistant

Natural language to SQL, with execution and automatic visualization, built on
LangGraph + litellm + Gradio.

## Layout

```
app.py                  Entry point (loads .env, launches the UI)
requirements.txt        Dependencies
.env                    Environment template (fill in your keys)
sql_assistant/
    config.py           Dataclasses and enums
    adapters.py         Database adapters, factory, schema cache
    parsing.py          JSON extraction and SQL validation
    state.py            Shared AgentState
    llm.py              litellm wrapper (with Ollama routing)
    nodes.py            Pipeline node implementations
    workflow.py         LangGraph assembly and routing
    orchestrator.py     End-to-end process_question
    ui.py               Gradio interface
```

## Setup

```bash
pip install -r requirements.txt
cp .env .env.local   # optional; edit values
# Fill in GROQ_API_KEY (or your provider key) in .env
python app.py
```

The app opens a local Gradio server. Set the database path and API key in the
UI's Advanced configuration, or via the `.env` file.

## Pipeline

`intent -> normalize -> table select -> verify -> refine schema -> plan ->
generate SQL -> execute -> visualization decision -> visualization generate`

## Notes

- SQLite databases are opened read-only. Generated SQL is validated to reject
  anything but a single `SELECT`/`WITH` statement, and a `LIMIT` is enforced.
- Set `SQL_MODEL` to an Ollama model name (e.g. `sqlcoder:7b-q5_1`) to route
  SQL generation to a local Ollama server at `OLLAMA_BASE_URL`.
