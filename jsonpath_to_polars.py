from typing import Union, List, Tuple, Optional, Dict, Set, Any, cast
import polars as pl
from polars import Expr
import re


def parse_predicate(predicate_str: str) -> Tuple[str, Set[str]]:
    """
    Parse a JSONPath predicate expression and extract referenced fields.

    Args:
        predicate_str: A predicate string from a JSONPath expression (inside [?...]).

    Returns:
        A tuple containing:
        - A transformed predicate string for polars evaluation
        - A set of field names referenced in the predicate
    """
    # Handle parentheses in predicates
    if predicate_str.startswith("(") and predicate_str.endswith(")"):
        predicate_str = predicate_str[1:-1]

    # Replace @.field with field for polars expression
    transformed = re.sub(r"@\.([a-zA-Z0-9_]+)", r"\1", predicate_str)

    # Extract all field references
    fields = set(re.findall(r"@\.([a-zA-Z0-9_]+)", predicate_str))

    # Convert comparison operators if needed
    transformed = transformed.replace("==", "=")

    return transformed, fields


def predicate_to_expr(predicate_str: str, return_expr: Expr) -> Expr:
    """
    Convert a JSONPath predicate to a polars when/then/otherwise expression.

    Args:
        predicate_str: The parsed predicate string.
        return_expr: The expression to return when the predicate is true.

    Returns:
        A polars expression that evaluates the predicate.
    """
    # Split the predicate to get the left side, operator, and right side
    match = re.match(r"([a-zA-Z0-9_]+)\s*([=<>!]+)\s*(.+)", predicate_str)
    if not match:
        raise ValueError(f"Cannot parse predicate: {predicate_str}")

    field, op, value = match.groups()

    # Convert string value to proper format
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]  # Remove quotes

    # Create the condition based on the operator
    if op == "=":
        condition = pl.col(field) == value
    elif op == "!=":
        condition = pl.col(field) != value
    elif op == ">":
        condition = pl.col(field) > float(value)
    elif op == "<":
        condition = pl.col(field) < float(value)
    elif op == ">=":
        condition = pl.col(field) >= float(value)
    elif op == "<=":
        condition = pl.col(field) <= float(value)
    else:
        raise ValueError(f"Unsupported operator: {op}")

    # Create the when/then/otherwise expression
    return pl.when(condition).then(return_expr).otherwise(pl.lit(None))


def jsonpath_to_polars(jsonpath: str) -> Expr:
    """
    Convert a JSONPath expression to a polars Expression.

    Args:
        jsonpath: A JSONPath string starting with '$.'.

    Returns:
        A polars Expression that extracts data according to the JSONPath.

    Raises:
        ValueError: If the JSONPath format is invalid or unsupported.
    """
    # Remove the leading "$."
    if not jsonpath.startswith("$."):
        raise ValueError("JSONPath must start with '$.'")

    path = jsonpath[2:]  # Remove "$."

    # Handle simple field access with no special characters
    if "." not in path and "[" not in path:
        return pl.col(path)

    # Handle simple nested field access (no wildcards, no array access)
    if "[" not in path and path.count(".") > 0:
        parts = path.split(".")
        root = parts[0]
        rest = ".".join(parts[1:])
        return pl.col(root).str.json_path_match(f"$.{rest}")

    # Special case for multiple array indices like $.matrix[0][1]
    multiple_array_pattern = r"^([a-zA-Z0-9_]+)(\[\d+\])+$"
    if re.match(multiple_array_pattern, path):
        root = path.split("[")[0]
        # Extract all indices
        indices = re.findall(r"\[(\d+)\]", path)
        indices_str = "[" + "][".join(indices) + "]"
        return pl.col(root).str.json_path_match(f"${indices_str}")

    # Handle array access patterns
    # 1. Simple array access like $.field.array[0]
    # 2. Nested object in array like $.field.array[0].property
    # 3. Negative array indices like $.field.array[-1]
    array_pattern = r"(.+?)(?:\[([-]?\d+)\])(.*)"
    array_match = re.match(array_pattern, path)
    if array_match:
        field_path, index, rest = array_match.groups()
        # Convert to int but no need to store since we use the string version in the expressions
        int(index)  # validate that index is a valid integer

        # Check for multiple array indices
        if rest and rest.startswith("[") and "]" in rest:
            # We have a pattern like $.matrix[0][1]
            # Extract the second index
            second_match = re.match(r"\[(\d+)\](.*)", rest)
            if second_match:
                second_index, remaining = second_match.groups()

                # Build the JSON path with both indices
                # For the root element
                if field_path.count(".") == 0:
                    return pl.col(field_path).str.json_path_match(f"$[{index}][{second_index}]{remaining}")
                else:
                    # For nested elements
                    parts = field_path.split(".")
                    root = parts[0]
                    nested = ".".join(parts[1:])
                    return pl.col(root).str.json_path_match(f"$.{nested}[{index}][{second_index}]{remaining}")

        # Split into root and nested path
        parts = field_path.split(".")
        root = parts[0]

        if len(parts) > 1:
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
        else:
            # Direct array access on the root field
            if rest:
                # Access object property after array element
                if rest.startswith("."):
                    rest = rest[1:]
                return pl.col(root).str.json_path_match(f"$[{index}].{rest}")
            else:
                # Just the array element
                return pl.col(root).str.json_path_match(f"$[{index}]")

    # Special case for array with predicate and field access
    if "[?(" in path and ")]" in path:
        # Extract the array field (before the predicate)
        array_path_parts = path.split("[?(")
        array_field = array_path_parts[0]

        # Extract the predicate and the return field
        rest = array_path_parts[1]
        predicate_end = rest.find(")]")
        predicate_str = rest[:predicate_end]

        # Extract return field if it exists after the predicate
        return_field = None
        if predicate_end + 2 < len(rest) and rest[predicate_end + 2] == ".":
            return_field = rest[predicate_end + 3 :]

        # Parse the predicate components
        # For JSONPath like $.inventory.items[?(@.price>100)].name
        # Extract the field part (price), operator (>), and value (100)
        pred_match = re.match(r"@\.([a-zA-Z0-9_]+)\s*(==|!=|>|<|>=|<=)\s*([^)&|]+)", predicate_str)

        if not pred_match:
            raise ValueError(f"Cannot parse predicate: {predicate_str}")

        field, op, value = pred_match.groups()

        # Clean up the value (remove quotes if needed)
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.isdigit():
            value = int(value)
        elif re.match(r"^-?\d+(\.\d+)?$", value):
            value = float(value)

        # Build the JSONPath expression to extract filtered data
        # First split the array field path to get the root column
        parts = array_field.split(".")
        root = parts[0]

        if len(parts) > 1:
            # We have a nested path like 'inventory.items'
            nested_part = ".".join(parts[1:])

            # For items with a predicate, construct a JSONPath expression
            # that directly filters using JSONPath syntax
            jsonpath_expr = f"$.{nested_part}[?(@.{field}{op}{value})]"

            if return_field:
                # If we want to extract a specific field from the filtered items
                jsonpath_expr += f".{return_field}"

            # Use JSONPath in Polars to do the filtering
            return pl.col(root).str.json_path_match(jsonpath_expr)
        else:
            # This is a direct array in the root column
            jsonpath_expr = f"$[?(@.{field}{op}{value})]"

            if return_field:
                jsonpath_expr += f".{return_field}"

            return pl.col(root).str.json_path_match(jsonpath_expr)

    # Tokenize the path for more complex processing
    tokens: List[Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]] = []
    current = ""
    i = 0

    # Parse the path into tokens (field names and array accesses)
    while i < len(path):
        if path[i] == ".":
            if current:
                tokens.append(("field", current))
                current = ""
            i += 1
        elif path[i] == "[":
            if current:
                tokens.append(("field", current))
                current = ""

            # Extract the array index, wildcard, or predicate
            i += 1

            # Check if this is a predicate expression
            if path[i] == "?":
                # Find the end of the predicate expression
                i += 1  # Skip the '?' character
                predicate_str = ""
                depth = 1  # To handle nested brackets

                while i < len(path) and depth > 0:
                    if path[i] == "]":
                        depth -= 1
                        if depth == 0:
                            break
                        predicate_str += path[i]
                    elif path[i] == "[":
                        depth += 1
                        predicate_str += path[i]
                    else:
                        predicate_str += path[i]
                    i += 1

                # Parse the predicate and collect field references
                parsed_pred, fields = parse_predicate(predicate_str)
                tokens.append(("predicate", {"expr": parsed_pred, "fields": fields}))
            else:
                # Regular array index or wildcard
                index_str = ""
                while i < len(path) and path[i] != "]":
                    index_str += path[i]
                    i += 1

                if index_str == "*":
                    tokens.append(("wildcard", None))
                else:
                    try:
                        tokens.append(("index", int(index_str)))
                    except ValueError:
                        # Handle other array accessor syntaxes if needed
                        tokens.append(("index_expr", index_str))

            i += 1  # Skip the closing bracket
        else:
            current += path[i]
            i += 1

    # Add the last token if there is one
    if current:
        tokens.append(("field", current))

    # Process tokens to build the polars expression
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
            # Check if the previous token was a wildcard or regular field
            if i > 0 and tokens[i - 1][0] == "wildcard":
                # After wildcard, we need to extract this field from each array element
                field_name = cast(str, token_value)
                expr = expr.str.json_path_match(f"$.{field_name}")
            else:
                # Regular nested field access
                prev_token_type = tokens[i - 1][0] if i > 0 else None
                field_name = cast(str, token_value)

                if prev_token_type == "field" and "." not in field_name:
                    # If this is a nested field access where a parent field accesses a child field
                    expr = expr.str.json_path_match(f"$.{field_name}")
                else:
                    # Simple field access
                    expr = expr.struct.field(field_name)  # type: ignore

        elif token_type == "index":
            # Array index access
            expr = expr.list.get(cast(int, token_value))  # type: ignore

        elif token_type == "wildcard":
            # Wildcard array access
            # Look ahead to see what fields we need to extract from array elements
            if i + 1 < len(tokens) and tokens[i + 1][0] == "field":
                next_field = cast(str, tokens[i + 1][1])
                expr = expr.str.json_decode(pl.List(pl.Struct([pl.Field(next_field, pl.String)])))
                i += 1  # Skip the next token as we've incorporated it
            else:
                # Generic array decode
                expr = expr.str.json_decode()

        elif token_type == "predicate":
            # Handle predicate expressions for filtering arrays
            pred_info = cast(Dict[str, Any], token_value)
            pred_expr = pred_info["expr"]
            fields = pred_info["fields"]

            # Store parent field name for context
            parent_field = None
            if i > 0 and tokens[i - 1][0] == "field":
                parent_field = cast(str, tokens[i - 1][1])

            # First decode the JSON array
            if not fields:
                # If no fields were extracted from the predicate, default to common ones
                struct_fields = [pl.Field("price", pl.Float64), pl.Field("name", pl.String)]
            else:
                struct_fields = [pl.Field(field, pl.String) for field in fields]

            # For predicates, we first need to decode the array to access its elements
            if i > 0 and tokens[i - 1][0] in ("field", "index"):
                # Regular predicate after a field or index
                expr = expr.str.json_decode(pl.List(pl.Struct(struct_fields)))

                # Now apply the filter
                # Look ahead to see what to return
                if i + 1 < len(tokens) and tokens[i + 1][0] == "field":
                    next_field = cast(str, tokens[i + 1][1])
                    return_expr = pl.col(next_field)
                    expr = predicate_to_expr(pred_expr, return_expr)
                    # Ensure we preserve the parent context in the expression
                    if parent_field:
                        expr = expr.alias(f"{parent_field}_filtered")
                    i += 1  # Skip the next token
                else:
                    # Return the whole matching objects
                    expr = predicate_to_expr(pred_expr, expr)

        i += 1

    return expr
