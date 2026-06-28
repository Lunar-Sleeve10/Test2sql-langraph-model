"""Configuration objects and enums."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ExecutionMode(Enum):
    DIRECT_SQL = "direct_sql"
    LIGHT_PLAN = "light_plan"
    FULL_PLAN = "full_plan"


@dataclass
class DatabaseConfig:
    db_type: str
    connection_string: str
    max_rows: int = 1000
    timeout: int = 30
    read_only: bool = True


@dataclass
class SystemConfig:
    llm_provider: str
    api_key: str
    model_name: str
    sql_model: str = "groq/llama-3.1-70b-versatile"
    judge_model: str = "groq/llama-3.1-8b-instant"
    temperature: float = 0.1
    max_clarifications: int = 3
    max_rows: int = 1000
    ollama_base_url: str = "http://localhost:11434"
    data_dictionary: Optional[Dict] = None
