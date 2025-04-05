"""Parsing utilities for JSONPath strings."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import polars as pl
from polars import Expr

from polar_express.parsing import predicate_parser


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
        token_value: The value of the current token (predicate string).

    Returns:
        The updated polars Expression.
    """
    # Handle predicate expressions for filtering arrays
    pred_expr = cast(str, token_value)

    # Extract fields from the predicate
    fields = predicate_parser.extract_fields_from_predicate(pred_expr)

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

        # Convert predicate to a polars expression
        pred_polars_expr = predicate_parser.convert_to_polars(pred_expr)

        # Now apply the filter
        # Look ahead to see what to return
        if idx + 1 < len(tokens) and tokens[idx + 1][0] == "field":
            next_field = cast(str, tokens[idx + 1][1])
            return_expr = pl.col(next_field)
            # Apply the filter and return the specified field
            expr = pl.when(pred_polars_expr).then(return_expr).otherwise(pl.lit(None))
            # Ensure we preserve the parent context in the expression
            if parent_field:
                expr = expr.alias(f"{parent_field}_filtered")
            return expr
        else:
            # Return the whole matching objects
            return pl.when(pred_polars_expr).then(expr).otherwise(pl.lit(None))

    return expr


def process_tokens(tokens: List[Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]]) -> Expr:
    """
    Process a list of tokens to build a polars expression.

    Always uses the first token for the column name. After that, we chain tokens together into
    a path to be traversed via `base_expr.str.json_path_match(reconstructed_path)`.

    Whenever an array wildcard or predicate is encountered, we stop the `json_path_match` traversal,
    determine the datatype of the list elements, then use `expr.str.json_decode(list_schema)`
    to handle the array wildcard or predicate.

    Args:
        tokens: The list of tokens to process.

    Returns:
        A polars Expression that represents the processed tokens.

    Raises:
        ValueError: If the tokens do not form a valid JSONPath.
    """
    if not tokens:
        return pl.lit(None)  # Empty path case

    # Start with the root element (always use the first token as column name)
    first_token = tokens[0]
    if first_token[0] == "field":
        base_expr = pl.col(cast(str, first_token[1]))
    else:
        # Handle unusual case where path starts with an array accessor
        raise ValueError("JSONPath must start with a field name after '$.'")

    # If we only have one token, return the base expression
    if len(tokens) == 1:
        return base_expr

    # Find the first array wildcard or predicate, if any
    split_idx = None
    for i, token in enumerate(tokens[1:], 1):  # Start from second token
        if token[0] in ("wildcard", "predicate"):
            split_idx = i
            break

    # If there's no wildcard or predicate, construct path and use json_path_match
    if split_idx is None:
        # Reconstruct path for all tokens after the first
        json_path = "$"
        for token in tokens[1:]:
            token_type, token_value = token
            if token_type == "field":
                json_path += f".{token_value}"
            elif token_type == "index":
                json_path += f"[{token_value}]"

        # Use json_path_match for simple path traversal
        return base_expr.str.json_path_match(json_path)

    # Handle the case with a wildcard or predicate
    # First, process everything before the wildcard/predicate with json_path_match
    if split_idx > 1:  # If there are tokens between root and wildcard/predicate
        json_path = "$"
        for token in tokens[1:split_idx]:
            token_type, token_value = token
            if token_type == "field":
                json_path += f".{token_value}"
            elif token_type == "index":
                json_path += f"[{token_value}]"

        # Apply the path up to the split point
        expr = base_expr.str.json_path_match(json_path)
    else:
        # No tokens between root and wildcard/predicate
        expr = base_expr

    # Handle the wildcard or predicate and everything after it
    split_token = tokens[split_idx]
    split_token_type = split_token[0]

    if split_token_type == "wildcard":
        # Collect field tokens that follow the wildcard to build schema
        field_tokens = []
        i = split_idx + 1
        while i < len(tokens) and tokens[i][0] == "field":
            field_tokens.append(tokens[i])
            i += 1

        if field_tokens:
            # Build schema for json_decode
            struct_fields = []
            for field_token in field_tokens:
                field_name = cast(str, field_token[1])
                struct_fields.append(pl.Field(field_name, pl.String))

            # Apply json_decode with list of structs schema
            schema = pl.List(pl.Struct(struct_fields))

            # Handle empty lists/null values
            return pl.when(expr.eq("[]").or_(expr.is_null())).then(pl.lit(None)).otherwise(expr.str.json_decode(schema))
        else:
            # Generic wildcard with no field access after
            return pl.when(expr.eq("[]").or_(expr.is_null())).then(pl.lit(None)).otherwise(expr.str.json_decode())

    elif split_token_type == "predicate":
        # Handle predicate expression
        pred_expr = cast(str, split_token[1])
        fields = predicate_parser.extract_fields_from_predicate(pred_expr)

        # Build schema for predicate fields
        struct_fields = [pl.Field(field, pl.String) for field in fields]
        schema = pl.List(pl.Struct(struct_fields))

        # Decode the JSON array first
        decoded_expr = (
            pl.when(expr.eq("[]").or_(expr.is_null())).then(pl.lit(None)).otherwise(expr.str.json_decode(schema))
        )

        # Convert predicate to a polars expression
        pred_polars_expr = predicate_parser.convert_to_polars(pred_expr)

        # Look ahead to see what to return after predicate filtering
        if split_idx + 1 < len(tokens) and tokens[split_idx + 1][0] == "field":
            next_field = cast(str, tokens[split_idx + 1][1])
            # The field to extract from filtered elements
            return_expr = pl.col(next_field)
            # Apply the filter and return the specified field
            return pl.when(pred_polars_expr).then(return_expr).otherwise(pl.lit(None))
        else:
            # Return the whole matching objects
            return pl.when(pred_polars_expr).then(decoded_expr).otherwise(pl.lit(None))

    # If we get here, we have a path we couldn't process
    # This is a fallback, but should be rare given the logic above
    return expr
