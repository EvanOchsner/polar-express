"""Parsing utilities for JSONPath strings."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import polars as pl
from polars import Expr


def validate_jsonpath(jsonpath: str) -> str:
    """
    Validate JSONPath and strip the leading '$.' or `$` (for initial array access like `$[0].foo`) prefix.

    Args:
        jsonpath: A JSONPath string starting with '$'.

    Returns:
        The JSONPath without the leading '$.' or `$` prefix.

    Raises:
        ValueError: If the JSONPath format is invalid.
    """
    if jsonpath.startswith("$."):
        return jsonpath[2:]  # Remove "$."
    elif jsonpath.startswith("$["):
        return jsonpath[1:]  # Remove "$"
    else:
        raise ValueError(f"Invalid JSONPath format: {jsonpath}")


def build_nested_schema(
    tokens: List[Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]], start_idx: int
) -> Tuple[pl.DataType, int]:
    """
    Build a nested schema for complex JSON paths with multiple array wildcards.

    Args:
        tokens: The list of tokens.
        start_idx: The starting index in the token list.

    Returns:
        A tuple of (schema, last_index_processed)
    """
    i = start_idx
    field_names: List[str] = []

    # Collect field names until we hit another wildcard or end of tokens
    while i < len(tokens) and tokens[i][0] == "field":
        field_names.append(cast(str, tokens[i][1]))
        i += 1

    # Check if we have another wildcard after collecting fields
    if i < len(tokens) and tokens[i][0] == "wildcard":
        # We hit another wildcard, which means we have nested arrays
        # Process tokens after this wildcard recursively
        inner_schema, last_idx = build_nested_schema(tokens, i + 1)

        # Wrap the inner schema in a List for the wildcard
        list_schema = pl.List(inner_schema)

        # Now wrap in Struct fields working backwards
        current_schema: pl.DataType = list_schema
        for field_name in reversed(field_names):
            current_schema = pl.Struct([pl.Field(field_name, current_schema)])

        return current_schema, last_idx
    else:
        # No more wildcards, build the terminal structure
        # Default to String for the innermost type
        innermost_type: pl.DataType = cast(pl.DataType, pl.String)

        # Build nested structure from inside out
        terminal_schema: pl.DataType = innermost_type
        for field_name in reversed(field_names):
            if terminal_schema == pl.String:
                # Innermost field
                terminal_schema = pl.Struct([pl.Field(field_name, pl.String)])
            else:
                # Wrap the previous structure
                terminal_schema = pl.Struct([pl.Field(field_name, terminal_schema)])

        return (
            terminal_schema,
            i - 1,
        )  # -1 because we want to return to the last field token


def process_field_token(
    expr: Expr, tokens: List[Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]], idx: int, token_value: Any
) -> Expr:
    """
    Process a field token and update the expression.

    Args:
        expr: The current expression.
        tokens: The list of all tokens.
        idx: The current token index.
        token_value: The value of the current token.

    Returns:
        The updated polars Expression.
    """
    # Check if the previous token was a wildcard or regular field
    if idx > 0 and tokens[idx - 1][0] == "wildcard":
        # After wildcard, we need to extract this field from each array element
        field_name = cast(str, token_value)
        return expr.str.json_path_match(f"$.{field_name}")
    else:
        # Regular nested field access
        prev_token_type = tokens[idx - 1][0] if idx > 0 else None
        field_name = cast(str, token_value)

        if prev_token_type == "field" and "." not in field_name:
            # If this is a nested field access where a parent field accesses a child field
            return expr.str.json_path_match(f"$.{field_name}")
        else:
            # Simple field access
            return expr.struct.field(field_name)  # type: ignore


def process_wildcard_token(
    expr: Expr, tokens: List[Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]], idx: int
) -> Expr:
    """
    Process a wildcard token and update the expression.

    Args:
        expr: The current expression.
        tokens: The list of all tokens.
        idx: The current token index.

    Returns:
        The updated polars Expression.
    """
    # Wildcard array access - we need to identify all nested fields after the wildcard
    if idx + 1 < len(tokens):
        # Collect all field tokens that follow the wildcard
        nested_fields: List[str] = []
        i = idx + 1

        while i < len(tokens) and tokens[i][0] == "field":
            nested_fields.append(cast(str, tokens[i][1]))
            i += 1

        if nested_fields:
            # We found fields after the wildcard - construct the nested structure
            # Start from the innermost field
            current_type = pl.String

            # Build the nested structure from inside out
            field_structs: List[pl.Field] = []
            for field_name in reversed(nested_fields):
                if not field_structs:
                    # Innermost field
                    field_structs.append(pl.Field(field_name, current_type))
                else:
                    # Wrap the previous structure
                    field_structs.append(pl.Field(field_name, pl.Struct([field_structs[-1]])))

            # The last item contains our complete structure
            # First check if the array is empty before trying to decode
            return (
                pl.when(
                    # Check if it's an empty list
                    expr.eq("[]").or_(expr.is_null())
                )
                .then(
                    # Return null for empty lists
                    pl.lit(None)
                )
                .otherwise(
                    # Only try to decode when it's not empty
                    expr.str.json_decode(pl.List(pl.Struct([field_structs[-1]])))
                )
            )

    # Generic array decode if no field tokens follow
    # Check if it's an empty list first
    return (
        pl.when(expr.eq("[]").or_(expr.is_null()))
        .then(pl.lit(None))
        .otherwise(
            # Only try to decode when it's not empty
            expr.str.json_decode()
        )
    )


def process_predicate_token(
    expr: Expr, tokens: List[Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]], idx: int, token_value: Any
) -> Expr:
    """
    Process a predicate token and update the expression.

    Args:
        expr: The current expression.
        tokens: The list of all tokens.
        idx: The current token index.
        token_value: The value of the current token.

    Returns:
        The updated polars Expression.
    """
    from polar_express.conversion.jsonpath_to_polars import simple_predicate_to_expr

    # Handle predicate expressions for filtering arrays
    pred_info = cast(Dict[str, Any], token_value)
    pred_expr = pred_info["expr"]
    fields = pred_info["fields"]

    # Store parent field name for context
    parent_field = None
    if idx > 0 and tokens[idx - 1][0] == "field":
        parent_field = cast(str, tokens[idx - 1][1])

    # First decode the JSON array
    struct_fields = [pl.Field(field, pl.String) for field in fields]

    # For predicates, we first need to decode the array to access its elements
    if idx > 0 and tokens[idx - 1][0] in ("field", "index"):
        # Regular predicate after a field or index
        expr = expr.str.json_decode(pl.List(pl.Struct(struct_fields)))

        # Now apply the filter
        # Look ahead to see what to return
        if idx + 1 < len(tokens) and tokens[idx + 1][0] == "field":
            next_field = cast(str, tokens[idx + 1][1])
            return_expr = pl.col(next_field)
            expr = simple_predicate_to_expr(pred_expr, return_expr)
            # Ensure we preserve the parent context in the expression
            if parent_field:
                expr = expr.alias(f"{parent_field}_filtered")
            return expr
        else:
            # Return the whole matching objects
            return simple_predicate_to_expr(pred_expr, expr)

    return expr


def process_tokens(tokens: List[Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]]) -> Expr:
    """
    Process a list of tokens to build a polars expression.

    Args:
        tokens: The list of tokens to process.

    Returns:
        A polars Expression that represents the processed tokens.

    Raises:
        ValueError: If the tokens do not form a valid JSONPath.
    """
    if not tokens:
        return pl.lit(None)  # Empty path case

    # Start with the root element
    first_token = tokens[0]
    if first_token[0] == "field":
        expr = pl.col(cast(str, first_token[1]))
    else:
        # Handle unusual case where path starts with an array accessor
        raise ValueError("JSONPath must start with a field name after '$.'")

    # Process the remaining tokens to build the chained expression
    i = 1
    while i < len(tokens):
        token = tokens[i]
        token_type = token[0]
        token_value = token[1]

        if token_type == "field":
            expr = process_field_token(expr, tokens, i, token_value)
        elif token_type == "index":
            # Handle indexed array access
            # If it's a numbered index followed by more complex path elements,
            # use json_path_match for the whole path
            if i > 0 and i + 1 < len(tokens) and (tokens[i + 1][0] == "wildcard" or tokens[i + 1][0] == "index"):
                # We have a complex path with multiple array accesses
                root_token = tokens[0]
                if root_token[0] == "field":
                    root_field = cast(str, root_token[1])

                    # Build JSONPath string for everything from the index onwards
                    json_path = "$"
                    for j in range(i, len(tokens)):
                        t = tokens[j]
                        if t[0] == "index":
                            json_path += f"[{t[1]}]"
                        elif t[0] == "wildcard":
                            json_path += "[*]"
                        elif t[0] == "field":
                            json_path += f".{t[1]}"

                    return pl.col(root_field).str.json_path_match(json_path)

            # Simple array index access
            expr = expr.list.get(cast(int, token_value))  # type: ignore
        elif token_type == "wildcard":
            # We need to handle complex wildcards with nested arrays differently
            next_wildcard_idx = next(
                (j for j in range(i + 1, len(tokens)) if tokens[j][0] == "wildcard"),
                None,
            )

            if next_wildcard_idx is not None:
                # We have multiple wildcards in the path, construct a custom schema
                schema, last_idx = build_nested_schema(tokens, i + 1)
                expr = expr.str.json_decode(pl.List(schema))
                i = last_idx
            else:
                # Simple wildcard handling
                expr = process_wildcard_token(expr, tokens, i)
                # Skip all field tokens that follow since they're processed in wildcard handling
                next_i = i + 1
                while next_i < len(tokens) and tokens[next_i][0] == "field":
                    next_i += 1
                i = next_i - 1  # -1 because we increment i at the end of the loop
        elif token_type == "predicate":
            expr = process_predicate_token(expr, tokens, i, token_value)
            # If we processed the next token as part of predicate handling
            if i + 1 < len(tokens) and tokens[i + 1][0] == "field":
                i += 1  # Skip the next token

        i += 1

    return expr
