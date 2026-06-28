"""Database adapters and schema cache."""

import os
import re
import sqlite3
import logging
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from .config import DatabaseConfig

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sanitize_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


class DatabaseAdapter(ABC):
    @abstractmethod
    def get_tables(self) -> List[str]: ...

    @abstractmethod
    def get_columns(self, table_name: str) -> List[str]: ...

    @abstractmethod
    def execute_query(self, query: str, limit: int) -> Tuple[List[Any], List[str]]: ...

    @abstractmethod
    def get_schema_info(self, tables: List[str]) -> str: ...

    @abstractmethod
    def get_full_schema(self) -> str: ...

    @abstractmethod
    def close(self) -> None: ...


class SQLiteAdapter(DatabaseAdapter):
    """SQLite adapter. Opens read-only by default for safety."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._lock = Lock()
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        if not os.path.exists(self.config.connection_string):
            raise FileNotFoundError(f"Database not found: {self.config.connection_string}")
        mode = "ro" if self.config.read_only else "rwc"
        self.conn = sqlite3.connect(
            f"file:{self.config.connection_string}?mode={mode}",
            uri=True,
            timeout=self.config.timeout,
            check_same_thread=False,
        )
        logger.info("Connected to SQLite: %s (mode=%s)", self.config.connection_string, mode)

    def get_tables(self) -> List[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            return [row[0] for row in cur.fetchall()]

    def get_columns(self, table_name: str) -> List[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(f"PRAGMA table_info({sanitize_identifier(table_name)});")
            return [row[1] for row in cur.fetchall()]

    def execute_query(self, query: str, limit: int) -> Tuple[List[Any], List[str]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(query)
            rows = cur.fetchmany(limit)
            cols = [d[0] for d in cur.description] if cur.description else []
            return rows, cols

    def get_schema_info(self, tables: List[str]) -> str:
        parts = []
        with self._lock:
            cur = self.conn.cursor()
            for table in tables:
                cur.execute(f"PRAGMA table_info({sanitize_identifier(table)});")
                cols = [f"{c[1]} {c[2]}" for c in cur.fetchall()]
                parts.append(f"{table}({', '.join(cols)})")
        return "\n".join(parts)

    def get_full_schema(self) -> str:
        return self.get_schema_info(self.get_tables())

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None


class GenericDatabaseAdapter(SQLiteAdapter):
    """Fallback adapter. Currently delegates to SQLite behaviour."""

    def _connect(self) -> None:
        logger.warning("Using SQLite fallback for db_type=%s", self.config.db_type)
        super()._connect()


def build_adapter(config: DatabaseConfig) -> DatabaseAdapter:
    """Factory: return the right adapter for a database config."""
    if config.db_type.lower() == "sqlite":
        return SQLiteAdapter(config)
    return GenericDatabaseAdapter(config)


class SchemaCache:
    _cache: Dict[str, str] = {}
    _lock = Lock()

    @classmethod
    def get_schema(cls, db_path: str, adapter: DatabaseAdapter) -> str:
        with cls._lock:
            if db_path not in cls._cache:
                logger.info("Building schema cache for: %s", db_path)
                cls._cache[db_path] = adapter.get_full_schema()
                logger.info("Schema cached: %d chars", len(cls._cache[db_path]))
            return cls._cache[db_path]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._cache.clear()
