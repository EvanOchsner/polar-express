"""Main conversion function for JSONPath to Polars."""

from typing import Any, List, Optional, Tuple

import polars as pl
from polars import Expr

from polar_express.conversion.handlers.array_handlers import (
    handle_array_access,
    handle_array_wildcard_access,
    handle_array_with_predicate,
    handle_multiple_array_indices,
    handle_multiple_array_patterns,
)
from polar_express.conversion.handlers.field_handlers import handle_simple_field_access
from polar_express.parsing.path_parser import (
    process_tokens,
    validate_jsonpath,
)
from polar_express.utils.tokens import tokenize_path


def comparison_to_expr(field: str, op: str, value: Any) -> Expr:
    """
    Convert a single comparison operation to a polars expression.

    Args:
        field: The field to compare
        op: The comparison operator (==, !=, >, <, >=, <=)
        value: The value to compare against

    Returns:
        A polars expression representing the comparison
    """
    # Create the condition based on the operator
    if op == "==":
        return pl.col(field).eq(value)
    elif op == "!=":
        return pl.col(field).ne(value)
    elif op == ">":
        return pl.col(field).gt(float(value))
    elif op == "<":
        return pl.col(field).lt(float(value))
    elif op == ">=":
        return pl.col(field).ge(float(value))
    elif op == "<=":
        return pl.col(field).le(float(value))
    else:
        raise ValueError(f"Unsupported operator: {op}")


def parse_predicate_expression(
    predicate_str: str,
) -> List[Tuple[str, str, Any, Optional[str]]]:
    """
    Parse a predicate expression into components.

    Supports all comparison operators (==, !=, >, <, >=, <=) with both string
    and numeric values, and supports both && (AND) and || (OR) operators between conditions.

    Args:
        predicate_str: The predicate string to parse.

    Returns:
        A list of tuples containing (field, operator, value, join_op).
        join_op can be "&&", "||", or None for the last condition.

    Raises:
        ValueError: If the predicate cannot be parsed.
    """
    # We'll parse predicate conditions with their joining operators
    result: List[Tuple[str, str, Any, Optional[str]]] = []

    # Track our position in the string
    pos = 0
    while pos < len(predicate_str):
        # Skip any leading whitespace
        while pos < len(predicate_str) and predicate_str[pos].isspace():
            pos += 1

        if pos >= len(predicate_str):
            break

        # Find the next comparison operator and extract the field and operator
        # Skip past the @. prefix
        if predicate_str[pos : pos + 2] == "@.":
            pos += 2
        else:
            raise ValueError(f"Expected @. prefix at position {pos}")

        # Extract the field name
        field_start = pos
        while pos < len(predicate_str) and (predicate_str[pos].isalnum() or predicate_str[pos] == "_"):
            pos += 1
        field = predicate_str[field_start:pos]

        # Skip whitespace before operator
        while pos < len(predicate_str) and predicate_str[pos].isspace():
            pos += 1

        # Extract the operator
        op_start = pos
        while pos < len(predicate_str) and predicate_str[pos] in "=!<>":
            pos += 1
        op = predicate_str[op_start:pos]

        # Skip whitespace after operator
        while pos < len(predicate_str) and predicate_str[pos].isspace():
            pos += 1

        # Extract the value (string or numeric)
        value: Any
        if pos < len(predicate_str) and predicate_str[pos] == '"':
            # String value
            pos += 1  # Skip opening quote
            value_start = pos
            while pos < len(predicate_str) and predicate_str[pos] != '"':
                pos += 1
            value = predicate_str[value_start:pos]
            pos += 1  # Skip closing quote
        else:
            # Numeric value or boolean
            value_start = pos

            # Check if it might be a boolean value
            if predicate_str[pos : pos + 4] == "true":
                value = True
                pos += 4
            elif predicate_str[pos : pos + 5] == "false":
                value = False
                pos += 5
            else:
                # Parse as numeric
                while pos < len(predicate_str) and (predicate_str[pos].isdigit() or predicate_str[pos] == "."):
                    pos += 1
                value_str = predicate_str[value_start:pos]
                if value_str:  # Make sure we have a non-empty string
                    # Convert the value to numeric type
                    value = float(value_str) if "." in value_str else int(value_str)
                else:
                    raise ValueError(f"Invalid value at position {pos} in predicate: {predicate_str}")

        # Skip whitespace after value
        while pos < len(predicate_str) and predicate_str[pos].isspace():
            pos += 1

        # Look for joining operator (&& or ||)
        join_op = None
        if pos + 1 < len(predicate_str):
            if predicate_str[pos : pos + 2] == "&&":
                join_op = "&&"
                pos += 2
            elif predicate_str[pos : pos + 2] == "||":
                join_op = "||"
                pos += 2

        # Add the parsed condition to our result
        result.append((field, op, value, join_op))

    if not result:
        raise ValueError(f"Could not parse predicate: {predicate_str}")

    return result


def simple_predicate_to_expr(predicate_str: str, return_expr: Expr) -> Expr:
    """
    Convert a JSONPath predicate expression to a polars expression.
    Supports complex expressions with AND (&&) and OR (||) operators.

    Args:
        predicate_str: The predicate JSONpath string; e.g. "@.foo == "bar" or "@.count > 1 && @.status == "active""
        return_expr: The expression to return when the predicate is true.

    Returns:
        A polars expression that evaluates the predicate.
    """
    # Parse the predicate into its components
    try:
        conditions = parse_predicate_expression(predicate_str)
    except ValueError as e:
        raise ValueError(f"Cannot parse predicate: {predicate_str} - {str(e)}")

    # Build the condition expression by combining the individual conditions
    if not conditions:
        raise ValueError(f"No valid conditions found in predicate: {predicate_str}")

    # Start with the first condition
    field, op, value, join_op = conditions[0]
    condition_expr = comparison_to_expr(field, op, value)

    # Add each subsequent condition, respecting the join operator
    for i in range(1, len(conditions)):
        field, op, value, join_op_next = conditions[i]
        next_expr = comparison_to_expr(field, op, value)

        # Use the join operator from the previous condition to connect this one
        prev_join_op = conditions[i - 1][3]
        if prev_join_op == "&&":
            condition_expr = condition_expr.and_(next_expr)
        elif prev_join_op == "||":
            condition_expr = condition_expr.or_(next_expr)
        else:
            # This shouldn't happen with valid input
            raise ValueError(f"Invalid join operator in predicate: {predicate_str}")

    # Create the when/then/otherwise expression
    return pl.when(condition_expr).then(return_expr).otherwise(pl.lit(None))


def jsonpath_to_polars(jsonpath: str) -> Expr:
    """
    Convert a JSONPath expression to a polars Expression.

    Args:
        jsonpath: A JSONPath string starting with '$'.

    Returns:
        A polars Expression that extracts data according to the JSONPath.

    Raises:
        ValueError: If the JSONPath format is invalid or unsupported.
    """
    # Validate and strip $ prefix
    path = validate_jsonpath(jsonpath)

    # Try each handler in sequence, returning the first successful result

    # Handle nested arrays with wildcards or predicates specially
    expr = handle_multiple_array_patterns(path)
    if expr is not None:
        return expr

    # Handle simple field access cases first
    expr = handle_simple_field_access(path)
    if expr is not None:
        return expr

    # Handle multiple array indices like $.matrix[0][1]
    expr = handle_multiple_array_indices(path)
    if expr is not None:
        return expr

    # Handle array wildcard patterns like $.foo.bar[*].baz with empty list handling
    expr = handle_array_wildcard_access(path)
    if expr is not None:
        return expr

    # Handle various array access patterns
    expr = handle_array_access(path)
    if expr is not None:
        return expr

    # Handle array with predicate
    expr = handle_array_with_predicate(path)
    if expr is not None:
        return expr

    # For more complex cases, tokenize and process
    tokens = tokenize_path(path)
    return process_tokens(tokens)
