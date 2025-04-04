"""Handlers for field access in JSONPath expressions."""

import re
from typing import Optional

import polars as pl
from polars import Expr


def handle_simple_field_access(path: str) -> Optional[Expr]:
    """
    Handle simple field access patterns without special characters.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression if the path matches this pattern, None otherwise.
    """
    # Simple field access with no special characters
    if "." not in path and "[" not in path:
        return pl.col(path)

    # Simple nested field access (no wildcards, no array access)
    if "[" not in path and path.count(".") > 0:
        parts = path.split(".")
        root = parts[0]
        rest = ".".join(parts[1:])
        return pl.col(root).str.json_path_match(f"$.{rest}")

    return None