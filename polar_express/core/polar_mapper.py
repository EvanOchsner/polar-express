"""PolarMapper class for building ETL pipelines with polars DataFrames."""

import json
from typing import Any, Dict, List, Optional, TextIO, cast

from polars import DataFrame, Expr


class PolarMapper:
    """
    A class that holds a sequence of computation steps (ETL) for a polars DataFrame.
    Each step is a polars operation with associated expressions.
    """

    def __init__(self) -> None:
        """Initialize an empty PolarMapper with no computation steps."""
        self.steps: List[Dict[str, Any]] = []

    def add_select_step(self, exprs: List[Expr]) -> "PolarMapper":
        """
        Add a select step to the computation pipeline.

        Args:
            exprs: List of polars expressions to select.

        Returns:
            Self for method chaining.
        """
        self.steps.append({"type": "select", "exprs": exprs})
        return self

    def add_with_columns_step(self, exprs: List[Expr]) -> "PolarMapper":
        """
        Add a with_columns step to the computation pipeline.

        Args:
            exprs: List of polars expressions to add as new columns.

        Returns:
            Self for method chaining.
        """
        self.steps.append({"type": "with_columns", "exprs": exprs})
        return self

    def add_filter_step(self, predicate_expr: Expr) -> "PolarMapper":
        """
        Add a filter step to the computation pipeline.

        Args:
            predicate_expr: A polars expression that evaluates to a boolean.

        Returns:
            Self for method chaining.
        """
        self.steps.append({"type": "filter", "expr": predicate_expr})
        return self

    def map(self, df: DataFrame) -> DataFrame:
        """
        Apply all computation steps to the input DataFrame.

        Args:
            df: Input polars DataFrame.

        Returns:
            The transformed DataFrame after all computation steps.
        """
        result = df

        for step in self.steps:
            step_type = step["type"]

            if step_type == "select":
                result = result.select(step["exprs"])
            elif step_type == "with_columns":
                result = result.with_columns(step["exprs"])
            elif step_type == "filter":
                result = result.filter(step["expr"])

        return result

    def to_string(self) -> str:
        """
        Create a string representation of the mapper.

        Returns:
            A string representation of all computation steps.
        """
        if not self.steps:
            return "PolarMapper with no steps"

        result = "PolarMapper with steps:\n"
        for i, step in enumerate(self.steps, 1):
            step_type = step["type"]

            if step_type == "select":
                exprs = [str(expr) for expr in cast(List[Expr], step["exprs"])]
                result += f"{i}. SELECT: {', '.join(exprs)}\n"
            elif step_type == "with_columns":
                exprs = [str(expr) for expr in cast(List[Expr], step["exprs"])]
                result += f"{i}. WITH_COLUMNS: {', '.join(exprs)}\n"
            elif step_type == "filter":
                result += f"{i}. FILTER: {step['expr']}\n"

        return result

    def __str__(self) -> str:
        """
        String representation of the PolarMapper.

        Returns:
            A concise string representation of the mapper.
        """
        return self.to_string()

    def __repr__(self) -> str:
        """
        Detailed representation of the PolarMapper.

        Returns:
            A detailed string representation of the mapper.
        """
        return f"PolarMapper(steps={len(self.steps)})"

    def to_json(self) -> str:
        """
        Create a JSON representation of the mapper.

        Returns:
            A JSON string representation of all computation steps.
        """
        serializable_steps = []

        for step in self.steps:
            serialized_step = {"type": step["type"]}

            if step["type"] in ["select", "with_columns"]:
                serialized_step["exprs"] = [str(expr) for expr in cast(List[Expr], step["exprs"])]
            elif step["type"] == "filter":
                serialized_step["expr"] = str(step["expr"])

            serializable_steps.append(serialized_step)

        return json.dumps({"steps": serializable_steps}, indent=2)

    def describe(self, output: Optional[TextIO] = None) -> Optional[str]:
        """
        Create a user-friendly description of the computation steps.

        Args:
            output: Optional output stream to write the description to.
                   If None, returns the description as a string.

        Returns:
            If output is None, returns a user-friendly description string.
            Otherwise, writes to the output stream and returns None.
        """
        if not self.steps:
            description = "This mapper performs no operations on the input DataFrame."
            if output:
                output.write(description)
                return None
            return description

        description = "This mapper performs the following operations on the input DataFrame:\n\n"

        for i, step in enumerate(self.steps, 1):
            step_type = step["type"]

            if step_type == "select":
                exprs = [str(expr) for expr in cast(List[Expr], step["exprs"])]
                expr_str = ", ".join(exprs)
                description += f"Step {i}: Select columns {expr_str}\n"
            elif step_type == "with_columns":
                exprs = [str(expr) for expr in cast(List[Expr], step["exprs"])]
                expr_str = ", ".join(exprs)
                description += f"Step {i}: Add columns {expr_str}\n"
            elif step_type == "filter":
                description += f"Step {i}: Filter rows where {step['expr']}\n"

            description += "\n"

        if output:
            output.write(description)
            return None

        return description
