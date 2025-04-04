"""Handlers for array access in JSONPath expressions."""

import re
from typing import List, Optional

import polars as pl
from polars import Expr

from polar_express.utils.tokens import tokenize_path, tokens_to_jsonpath


def handle_multiple_array_indices(path: str) -> Optional[Expr]:
    """
    Handle multiple array indices pattern like $.matrix[0][1].

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression if the path matches this pattern, None otherwise.
    """
    multiple_array_pattern = r"^([a-zA-Z0-9_]+)(\[\d+\])+$"
    if re.match(multiple_array_pattern, path):
        root = path.split("[")[0]
        # Extract all indices
        indices = re.findall(r"\[(\d+)\]", path)
        indices_str = "[" + "][".join(indices) + "]"
        return pl.col(root).str.json_path_match(f"${indices_str}")

    return None


def handle_direct_array_access(root: str, index: str, rest: str) -> Expr:
    """
    Handle direct array access on the root field.

    Args:
        root: The root column name.
        index: The array index.
        rest: The remaining path after the array index.

    Returns:
        A polars Expression handling this pattern.
    """
    if rest:
        # Access object property after array element
        if rest.startswith("."):
            rest = rest[1:]
        return pl.col(root).str.json_path_match(f"$[{index}].{rest}")
    else:
        # Just the array element
        return pl.col(root).str.json_path_match(f"$[{index}]")


def handle_nested_array_access(root: str, parts: List[str], index: str, rest: str) -> Expr:
    """
    Handle nested array access like $.user_data.accounts[0].

    Args:
        root: The root column name.
        parts: The split parts of the field path.
        index: The array index.
        rest: The remaining path after the array index.

    Returns:
        A polars Expression handling this pattern.
    """
    # There's a path before the array access
    nested = ".".join(parts[1:])

    if rest:
        # We have a pattern like $.user_data.accounts[0].balance
        # Remove leading dot from rest if present
        if rest.startswith("."):
            rest = rest[1:]

        # Extract nested object from array element
        return pl.col(root).str.json_path_match(f"$.{nested}[{index}].{rest}")
    else:
        # Just array access without nested field after it
        return pl.col(root).str.json_path_match(f"$.{nested}[{index}]")


def handle_double_array_index(field_path: str, first_index: str, rest: str) -> Optional[Expr]:
    """
    Handle double array index pattern like $.matrix[0][1].

    Args:
        field_path: The path before the first array index.
        first_index: The first array index.
        rest: The remaining path after the first array index.

    Returns:
        A polars Expression handling this pattern.
    """
    second_match = re.match(r"\[(\d+)\](.*)", rest)
    if not second_match:
        return None

    second_index, remaining = second_match.groups()

    # Build the JSON path with both indices
    if field_path.count(".") == 0:
        # For the root element
        return pl.col(field_path).str.json_path_match(f"$[{first_index}][{second_index}]{remaining}")
    else:
        # For nested elements
        parts = field_path.split(".")
        root = parts[0]
        nested = ".".join(parts[1:])
        return pl.col(root).str.json_path_match(f"$.{nested}[{first_index}][{second_index}]{remaining}")


def handle_array_access(path: str) -> Optional[Expr]:
    """
    Handle array access patterns:
    1. Simple array access like $.field.array[0]
    2. Nested object in array like $.field.array[0].property
    3. Negative array indices like $.field.array[-1]

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression if the path matches this pattern, None otherwise.
    """
    array_pattern = r"(.+?)(?:\[([-]?\d+)\])(.*)"
    array_match = re.match(array_pattern, path)

    if not array_match:
        return None

    field_path, index, rest = array_match.groups()
    # Validate that index is a valid integer
    int(index)

    # Check for multiple array indices
    if rest and rest.startswith("[") and "]" in rest:
        return handle_double_array_index(field_path, index, rest)

    # Split into root and nested path
    parts = field_path.split(".")
    root = parts[0]

    if len(parts) > 1:
        return handle_nested_array_access(root, parts, index, rest)
    else:
        return handle_direct_array_access(root, index, rest)


def handle_array_wildcard_access(path: str) -> Optional[Expr]:
    """
    Handle array wildcard access patterns like $.foo.bar[*].baz,
    gracefully handling empty lists.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression if the path matches this pattern, None otherwise.
    """
    if "[*]" not in path:
        return None

    # Split the path at the wildcard
    parts = path.split("[*]")
    field_path = parts[0]  # e.g., "foo.bar"
    rest_path = parts[1] if len(parts) > 1 else ""  # e.g., ".baz"

    # Parse the field path before wildcard into tokens
    tokens = tokenize_path(field_path)

    # Extract the root column name from the first token
    if not tokens:
        return None

    root = ""
    if tokens[0][0] == "field":
        root = str(tokens[0][1])
    else:
        return None  # First token must be a field name

    # Determine base expression based on token count
    if len(tokens) == 1:
        # Direct array access on root column (e.g., "users[*]")
        base_expr = pl.col(root)
    else:
        # Complex path before wildcard (e.g., "users.profiles[0].data[*]")
        # Convert tokens to a proper JSONPath string (excluding the root field)
        rest_tokens = tokens[1:]
        if rest_tokens:
            # Create a JSONPath string from the tokens after the root
            json_path = tokens_to_jsonpath(rest_tokens)
            base_expr = pl.col(root).str.json_path_match(json_path)
        else:
            base_expr = pl.col(root)

    if rest_path == "":  # No trailing field, decode whole struct elements inside list
        decoded_expr = base_expr.str.json_decode(infer_schema_length=None)
    else:  # Only decode the trailing field portion of each element inside list
        rest_path_clean = rest_path.lstrip(".")  # Remove leading dot if present
        nested_parts = rest_path_clean.split(".")

        # Build the schema for list elements and decode
        current_schema: pl.DataType = pl.Utf8()
        for field in reversed(nested_parts):
            current_schema = pl.Struct([pl.Field(field, current_schema)])
        decoded_expr = base_expr.str.json_decode(pl.List(current_schema))

        # Build the nested field access chain
        field_expr = pl.element()
        for field in nested_parts:
            field_expr = field_expr.struct.field(field)

        # Apply the field access to each element in the list
        decoded_expr = decoded_expr.list.eval(field_expr)

    return pl.when(base_expr.eq("[]").or_(base_expr.is_null())).then(pl.lit(None)).otherwise(decoded_expr)


def has_multiple_array_patterns(path: str) -> bool:
    """
    Determines if a JSONPath has multiple array wildcards or predicates.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        True if the path contains nested arrays with wildcards or predicates, False otherwise.
    """
    # Count wildcards [*]
    wildcard_count = path.count("[*]")

    # Count predicates [?(...)
    predicate_count = path.count("[?(")

    # If we have multiple wildcards or predicates, or both a wildcard and predicate
    return (wildcard_count + predicate_count) > 1 or (wildcard_count >= 1 and predicate_count >= 1)


def handle_multiple_array_patterns(path: str) -> Optional[Expr]:
    """
    Handle JSONPaths that contain nested arrays with wildcards or predicates.
    These are hard to process fully, so we convert to string at the first array.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression that returns the JSON as a string at the point of the first wildcard/predicate array.
    """
    if not has_multiple_array_patterns(path):
        return None

    # Find the first array with wildcard or predicate
    wildcard_pos = path.find("[*]")
    predicate_pos = path.find("[?(")

    # Determine which comes first
    first_special_array_pos = float("inf")
    if wildcard_pos >= 0:
        first_special_array_pos = wildcard_pos
    if predicate_pos >= 0 and predicate_pos < first_special_array_pos:
        first_special_array_pos = predicate_pos

    # If neither was found, this shouldn't happen as we already checked for nested arrays
    if first_special_array_pos == float("inf"):
        return None

    # Find the path up to the first special array
    path_before = path[: int(first_special_array_pos)]

    # Handle the case where we have an indexed array access before a wildcard
    # Example: $.schools[0].classes[*].students[*].grade
    index_match = re.search(r"\[(\d+)\]", path_before)
    if index_match:
        # Extract everything before the indexed array bracket
        bracket_pos = path_before.find(f"[{index_match.group(1)}]")
        root_field = path_before[:bracket_pos]

        # Extract everything after the indexed array bracket
        rest_path = path_before[bracket_pos + len(index_match.group(0)):]
        if rest_path.startswith("."):
            rest_path = rest_path[1:]  # Remove leading dot

        # Extract the column name (first part of the path before any dots)
        if "." in root_field:
            parts = root_field.split(".")
            column_name = parts[0]
            nested_path = ".".join(parts[1:])

            # Build the expression using proper column name and nested paths
            # For example: pl.col("education_data").str.json_path_match("$.schools[0].classes").cast(pl.Utf8)
            return (
                pl.col(column_name)
                .str.json_path_match(f"$.{nested_path}[{index_match.group(1)}]{rest_path and f'.{rest_path}' or ''}")
                .cast(pl.Utf8)
            )
        else:
            # If there are no dots, then root_field is the column name
            # For example: pl.col("schools").str.json_path_match("$[0].classes").cast(pl.Utf8)
            return (
                pl.col(root_field)
                .str.json_path_match(f"$[{index_match.group(1)}]{rest_path and f'.{rest_path}' or ''}")
                .cast(pl.Utf8)
            )

    # Handle regular case (no indexed array before wildcard)
    # Get the root column
    if "." in path_before:
        parts = path_before.split(".")
        root = parts[0]
        rest_before = ".".join(parts[1:])

        # Use json_path_match to get to the array, then cast to string to keep the raw JSON
        # This gives us the JSONPath up to the array but not including the wildcard/predicate
        # We'll get the array as a string, which will include all nested structures
        return pl.col(root).str.json_path_match(f"$.{rest_before}").cast(pl.Utf8)
    else:
        # If there's no dot, the root is the full path_before
        # Cast the array field to string to get the raw JSON representation
        return pl.col(path_before).cast(pl.Utf8)


def handle_array_with_predicate(path: str) -> Optional[Expr]:
    """
    Handle array with predicate like $.items[?(@.field1 == "x1" && @.field2 == "x2")].name.
    Supports all comparison operators (==, !=, >, <, >=, <=) joined by AND (&&) and OR (||) operators.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression if the path matches this pattern, None otherwise.
    """
    from polar_express.parsing import predicate_parser

    if "[?(" not in path or ")]" not in path:
        return None

    # Extract the array field (before the predicate)
    array_path_parts = path.split("[?(")
    array_field = array_path_parts[0]

    # Extract the predicate, predicate fields and the return field
    rest = array_path_parts[1]
    predicate_end = rest.find(")]")
    predicate_str = rest[:predicate_end]
    predicate_fields = predicate_parser.extract_fields_from_predicate(predicate_str)
    return_field = None
    if predicate_end + 2 < len(rest) and rest[predicate_end + 2] == ".":
        return_field = rest[predicate_end + 3:]

    # Parse the field path before predicate into tokens
    tokens = tokenize_path(array_field)

    # Extract the root column name from the first token
    if not tokens:
        return None

    root = ""
    if tokens[0][0] == "field":
        root = str(tokens[0][1])
    else:
        return None  # First token must be a field name

    # Build a dtype for the encoded list we are filtering
    struct_fields = predicate_fields
    if return_field:
        struct_fields.append(return_field)
    encoded_list_dtype = pl.List(pl.Struct([pl.Field(f, pl.Utf8) for f in struct_fields]))

    # Convert the predicate into a pl.Expr
    predicate_expr = predicate_parser.convert_to_polars(predicate_str)

    # Determine base expression based on token count
    if len(tokens) == 1:
        # Direct array access on root column (e.g., "users[?(...))]")
        base_expr = pl.col(root)
    else:
        # Complex path before predicate (e.g., "users.profiles[0].data[?(...))]")
        # Convert tokens to a proper JSONPath string (excluding the root field)
        rest_tokens = tokens[1:]
        if rest_tokens:
            # Create a JSONPath string from the tokens after the root
            json_path = tokens_to_jsonpath(rest_tokens)
            base_expr = pl.col(root).str.json_path_match(json_path)
        else:
            base_expr = pl.col(root)

    if return_field:  # We just return this field for the matching items
        return_val = pl.element().struct.field(return_field)
    else:  # We return a struct with all fields referenced in the predicate for the matching items
        return_val = pl.element()

    return (
        pl.when(base_expr.eq("[]").or_(base_expr.is_null()))
        .then(pl.lit(None))
        .otherwise(
            base_expr.str.json_decode(encoded_list_dtype)
            .list.eval(pl.when(predicate_expr).then(return_val).otherwise(pl.lit(None)))
            .list.drop_nulls()
        )
    )
