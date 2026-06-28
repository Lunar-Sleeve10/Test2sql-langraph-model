"""LLM response parsing and SQL validation."""

import re
import json
from typing import Dict, Optional, Tuple


class LLMResponseParser:
    @staticmethod
    def extract_json(response_text: str) -> Optional[Dict]:
        """Extract the first balanced JSON object from text."""
        if not response_text or not isinstance(response_text, str):
            return None
        text = re.sub(r"```(?:json)?\s*|\s*```", "", response_text).strip()

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def extract_sql(response_text: str) -> str:
        if not response_text:
            return ""
        match = re.search(r"```(?:sql)?\s*(.*?)```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            if len(sql) > 10:
                return sql.rstrip(";").strip() + ";"
        match = re.search(r"((?:WITH|SELECT)\s+.*?;)", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


class SQLValidator:
    FORBIDDEN = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "ATTACH", "DETACH", "CREATE", "REPLACE", "PRAGMA", "VACUUM",
        "REINDEX", "GRANT", "REVOKE",
    ]

    @classmethod
    def validate(cls, sql: str) -> Tuple[bool, str]:
        if not sql or not sql.strip():
            return False, "Empty query"

        # Strip string/identifier literals so keywords inside them are ignored.
        cleaned = re.sub(r"'[^']*'|\"[^\"]*\"", "", sql)

        # Reject stacked statements (only a trailing semicolon allowed).
        if cleaned.rstrip().rstrip(";").count(";") > 0:
            return False, "Multiple statements are not allowed"

        upper = cleaned.upper()
        for kw in cls.FORBIDDEN:
            if re.search(rf"\b{kw}\b", upper):
                return False, f"Forbidden keyword: {kw}"

        if not re.search(r"\b(SELECT|WITH)\b", upper):
            return False, "Query must be a SELECT"

        return True, ""

    @staticmethod
    def enforce_limit(sql: str, max_rows: int) -> str:
        """Append a LIMIT to a query that lacks one."""
        body = sql.rstrip().rstrip(";")
        if re.search(r"\bLIMIT\b", body, re.IGNORECASE):
            return body + ";"
        return f"{body}\nLIMIT {max_rows};"
