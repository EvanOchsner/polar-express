from typing import Union, List, Tuple, Optional, Dict, Set, Any, cast
import polars as pl
from polars import Expr
import re

# Type for tokens produced during parsing
Token = Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]


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

    return transformed, fields


def simple_predicate_to_expr(predicate_str: str, return_expr: Expr) -> Expr:
    """
    Convert a JSONPath expression involving a single field comparison to a polars when/then/otherwise expression.

    Args:
        predicate_str: The simple predicate JSONpath string; e.g. "@.foo == "bar" or "@.count > 1"
        return_expr: The expression to return when the predicate is true.

    Returns:
        A polars expression that evaluates the simple predicate.
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
    if op == "==":
        condition = pl.col(field).eq(value)
    elif op == "!=":
        condition = pl.col(field).ne(value)
    elif op == ">":
        condition = pl.col(field).gt(float(value))
    elif op == "<":
        condition = pl.col(field).lt(float(value))
    elif op == ">=":
        condition = pl.col(field).ge(float(value))
    elif op == "<=":
        condition = pl.col(field).le(float(value))
    else:
        raise ValueError(f"Unsupported operator: {op}")

    # Create the when/then/otherwise expression
    return pl.when(condition).then(return_expr).otherwise(pl.lit(None))


def validate_jsonpath(jsonpath: str) -> str:
    """
    Validate JSONPath and strip the leading '$.' prefix.

    Args:
        jsonpath: A JSONPath string starting with '$.'.

    Returns:
        The JSONPath without the leading '$.' prefix.

    Raises:
        ValueError: If the JSONPath format is invalid.
    """
    if not jsonpath.startswith("$."):
        raise ValueError("JSONPath must start with '$.'")

    return jsonpath[2:]  # Remove "$."


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


def handle_double_array_index(
    field_path: str, first_index: str, rest: str
) -> Optional[Expr]:
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
        return pl.col(field_path).str.json_path_match(
            f"$[{first_index}][{second_index}]{remaining}"
        )
    else:
        # For nested elements
        parts = field_path.split(".")
        root = parts[0]
        nested = ".".join(parts[1:])
        return pl.col(root).str.json_path_match(
            f"$.{nested}[{first_index}][{second_index}]{remaining}"
        )


def handle_nested_array_access(
    root: str, parts: List[str], index: str, rest: str
) -> Expr:
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


def parse_predicate_expression(predicate_str: str) -> List[Tuple[str, str, Any]]:
    """
    Parse a predicate expression into components.
    
    Supports all comparison operators (==, !=, >, <, >=, <=) with both string 
    and numeric values, but only allows && (AND) operators between conditions.

    Args:
        predicate_str: The predicate string to parse.

    Returns:
        A list of tuples containing (field, operator, value).

    Raises:
        ValueError: If the predicate cannot be parsed or contains unsupported operators like OR.
    """
    # Check if there's an OR operator, which we don't support
    if "||" in predicate_str:
        raise ValueError("OR operators (||) are not supported in predicates.")
        
    # Split the predicate by && operator
    parts = predicate_str.split("&&")
    result = []
    
    # Parse each individual condition
    for condition in parts:
        condition = condition.strip()
        
        # Match pattern for any comparison with a string value: @.field op "value"
        string_match = re.match(r"@\.([a-zA-Z0-9_]+)\s*(==|!=|>|<|>=|<=)\s*\"([^\"]+)\"", condition)
        if string_match:
            field, op, value = string_match.groups()
            result.append((field, op, value.strip()))
            continue
            
        # Match pattern for any comparison with a numeric value: @.field op 100
        numeric_match = re.match(r"@\.([a-zA-Z0-9_]+)\s*(==|!=|>|<|>=|<=)\s*(\d+(?:\.\d+)?)", condition)
        if numeric_match:
            field, op, value = numeric_match.groups()
            # Convert the value to numeric type
            value = float(value) if "." in value else int(value)
            result.append((field, op, value))
            continue
            
        # If we get here, the pattern is not supported
        raise ValueError(f"Unsupported predicate condition: {condition}")
    
    return result


def handle_array_with_predicate(path: str) -> Optional[Expr]:
    """
    Handle array with predicate like $.items[?(@.field1 == "x1" && @.field2 == "x2")].name.
    Supports all comparison operators (==, !=, >, <, >=, <=) joined by AND operators.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression if the path matches this pattern, None otherwise.
    """
    if "[?(" not in path or ")]" not in path:
        return None

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

    try:
        # Parse the predicate components - now returns a list of conditions
        predicate_conditions = parse_predicate_expression(predicate_str)
    except ValueError:
        # If parsing fails, return None to let other handlers try
        return None

    # Build a JSONPath expression with all the conditions joined by AND
    jsonpath_predicate_parts = []
    
    for field, op, value in predicate_conditions:
        # Format the value based on its type
        if isinstance(value, str):
            # For string values, use double quotes in JSONPath
            formatted_value = f'"{value}"'
        else:
            # For numeric values, no quotes
            formatted_value = str(value)
        
        # Add the formatted condition
        jsonpath_predicate_parts.append(f"@.{field}{op}{formatted_value}")
    
    # Join all conditions with AND
    jsonpath_predicate = " && ".join(jsonpath_predicate_parts)

    # Build the JSONPath expression to extract filtered data
    # First split the array field path to get the root column
    parts = array_field.split(".")
    root = parts[0]

    if len(parts) > 1:
        # We have a nested path like 'inventory.items'
        nested_part = ".".join(parts[1:])

        # Construct a JSONPath expression that directly filters using JSONPath syntax
        jsonpath_expr = f"$.{nested_part}[?({jsonpath_predicate})]"

        if return_field:
            # If we want to extract a specific field from the filtered items
            jsonpath_expr += f".{return_field}"

        # Get the expression that would extract the array itself
        array_expr = pl.col(root).str.json_path_match(f"$.{nested_part}")

        # Check if the array is empty before trying to filter
        return pl.when(
            array_expr.eq("[]").or_(array_expr.is_null())
        ).then(
            pl.lit(None)
        ).otherwise(
            pl.col(root).str.json_path_match(jsonpath_expr)
        )
    else:
        # This is a direct array in the root column
        jsonpath_expr = f"$[?({jsonpath_predicate})]"

        if return_field:
            jsonpath_expr += f".{return_field}"

        # Check if the array is empty before trying to filter
        return pl.when(
            pl.col(root).eq("[]").or_(pl.col(root).is_null())
        ).then(
            pl.lit(None)
        ).otherwise(
            pl.col(root).str.json_path_match(jsonpath_expr)
        )


def tokenize_path(path: str) -> List[Token]:
    """
    Tokenize a JSONPath string into a list of tokens.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A list of tokens representing the path components.
    """
    tokens: List[Token] = []
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
                tokens.append(handle_predicate_token(path, i))
                # Find the end of the predicate expression and update i
                depth = 1  # To handle nested brackets
                i += 1  # Skip the '?' character

                while i < len(path) and depth > 0:
                    if path[i] == "]":
                        depth -= 1
                        if depth == 0:
                            break
                    elif path[i] == "[":
                        depth += 1
                    i += 1

                i += 1  # Skip the closing bracket
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

    return tokens


def handle_predicate_token(path: str, start_idx: int) -> Token:
    """
    Extract and parse a predicate token from the path.

    Args:
        path: The JSONPath string.
        start_idx: The starting index of the predicate in the path.

    Returns:
        A predicate token.
    """
    # Find the end of the predicate expression
    i = start_idx + 1  # Skip the '?' character
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
    return ("predicate", {"expr": parsed_pred, "fields": fields})


def build_nested_schema(tokens: List[Token], start_idx: int) -> Tuple[pl.DataType, int]:
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


def process_tokens(tokens: List[Token]) -> Expr:
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
            if (
                i > 0
                and i + 1 < len(tokens)
                and (tokens[i + 1][0] == "wildcard" or tokens[i + 1][0] == "index")
            ):
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


def process_field_token(
    expr: Expr, tokens: List[Token], idx: int, token_value: Any
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


def process_wildcard_token(expr: Expr, tokens: List[Token], idx: int) -> Expr:
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
                    field_structs.append(
                        pl.Field(field_name, pl.Struct([field_structs[-1]]))
                    )

            # The last item contains our complete structure
            # First check if the array is empty before trying to decode
            return pl.when(
                # Check if it's an empty list
                expr.eq("[]").or_(expr.is_null())
            ).then(
                # Return null for empty lists
                pl.lit(None)
            ).otherwise(
                # Only try to decode when it's not empty
                expr.str.json_decode(pl.List(pl.Struct([field_structs[-1]])))
            )

    # Generic array decode if no field tokens follow
    # Check if it's an empty list first
    return pl.when(
        expr.eq("[]").or_(expr.is_null())
    ).then(
        pl.lit(None)
    ).otherwise(
        # Only try to decode when it's not empty
        expr.str.json_decode()
    )


def process_predicate_token(
    expr: Expr, tokens: List[Token], idx: int, token_value: Any
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
    # Handle predicate expressions for filtering arrays
    pred_info = cast(Dict[str, Any], token_value)
    pred_expr = pred_info["expr"]
    fields = pred_info["fields"]

    # Store parent field name for context
    parent_field = None
    if idx > 0 and tokens[idx - 1][0] == "field":
        parent_field = cast(str, tokens[idx - 1][1])

    # First decode the JSON array
    if not fields:
        # If no fields were extracted from the predicate, default to common ones
        struct_fields = [pl.Field("price", pl.Float64), pl.Field("name", pl.String)]
    else:
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


def handle_array_wildcard_access(path: str) -> Optional[Expr]:
    """
    Handle array wildcard access patterns like $.foo.bar[*].baz,
    gracefully handling empty lists.
    
    Args:
        path: The JSONPath without the leading '$.' prefix.
        
    Returns:
        A polars Expression that safely extracts data from arrays.
    """
    if "[*]" not in path:
        return None
        
    # Split the path at the wildcard
    parts = path.split("[*]")
    field_path = parts[0]  # e.g., "foo.bar"
    rest_path = parts[1] if len(parts) > 1 else ""  # e.g., ".baz"
    
    # Split field path into components
    field_parts = field_path.split(".")
    root = field_parts[0]  # The root column name
    
    if len(field_parts) > 1:
        # Nested field before wildcard, e.g., "foo.bar"
        nested_path = ".".join(field_parts[1:])
        
        # First, extract the array itself
        array_expr = pl.col(root).str.json_path_match(f"$.{nested_path}")
        
        # Then check if it's an empty array before trying to decode
        if rest_path:
            # If there's a nested field after the wildcard
            rest_path = rest_path.lstrip(".")  # Remove leading dot if present
            
            # Parse nested fields for proper schema construction
            nested_parts = rest_path.split(".")
            
            # Build nested schema from inside out
            current_schema: pl.DataType = pl.String()
            
            for field in reversed(nested_parts):
                current_schema = pl.Struct([pl.Field(field, current_schema)])
            
            return pl.when(
                # Check if it's an empty list
                array_expr.eq("[]").or_(array_expr.is_null())
            ).then(
                # Return null for empty lists
                pl.lit(None)
            ).otherwise(
                # Only try to decode when it's not empty
                array_expr.str.json_decode(pl.List(current_schema))
            )
        else:
            # Just the array itself
            return pl.when(
                # Check if it's an empty list
                array_expr.eq("[]").or_(array_expr.is_null())
            ).then(
                # Return null for empty lists
                pl.lit(None)
            ).otherwise(
                # Only try to decode when it's not empty
                # Let Polars infer the schema from the JSON data
                array_expr.str.json_decode(infer_schema_length=None)
            )
    else:
        # Direct array access on root column, e.g., "items[*].price"
        if rest_path:
            # With nested field after wildcard
            rest_path = rest_path.lstrip(".")  # Remove leading dot if present
            
            # Parse nested fields for proper schema construction
            nested_parts = rest_path.split(".")
            
            # Build nested schema from inside out
            curr_schema: pl.DataType = pl.String()
            
            for field in reversed(nested_parts):
                curr_schema = pl.Struct([pl.Field(field, curr_schema)])
            
            return pl.when(
                # Check if it's an empty list
                pl.col(root).eq("[]").or_(pl.col(root).is_null())
            ).then(
                # Return null for empty lists
                pl.lit(None)
            ).otherwise(
                # Only try to decode when it's not empty
                pl.col(root).str.json_decode(pl.List(curr_schema))
            )
        else:
            # Just the array itself
            return pl.when(
                # Check if it's an empty list
                pl.col(root).eq("[]").or_(pl.col(root).is_null())
            ).then(
                # Return null for empty lists
                pl.lit(None)
            ).otherwise(
                # Only try to decode when it's not empty
                # Let Polars infer the schema from the JSON data
                pl.col(root).str.json_decode(infer_schema_length=None)
            )


def has_nested_array_wildcards_or_predicates(path: str) -> bool:
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
    return (wildcard_count + predicate_count) > 1 or (
        wildcard_count >= 1 and predicate_count >= 1
    )


def handle_nested_arrays_special(path: str) -> Optional[Expr]:
    """
    Handle JSONPaths that contain nested arrays with wildcards or predicates.
    These are hard to process fully, so we convert to string at the first array.

    Args:
        path: The JSONPath without the leading '$.' prefix.

    Returns:
        A polars Expression that returns the JSON as a string at the point of the first wildcard/predicate array.
    """
    if not has_nested_array_wildcards_or_predicates(path):
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
        rest_path = path_before[bracket_pos + len(index_match.group(0)) :]
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
                .str.json_path_match(
                    f"$.{nested_path}[{index_match.group(1)}]{rest_path and f'.{rest_path}' or ''}"
                )
                .cast(pl.Utf8)
            )
        else:
            # If there are no dots, then root_field is the column name
            # For example: pl.col("schools").str.json_path_match("$[0].classes").cast(pl.Utf8)
            return (
                pl.col(root_field)
                .str.json_path_match(
                    f"$[{index_match.group(1)}]{rest_path and f'.{rest_path}' or ''}"
                )
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
    # Validate and strip the leading "$."
    path = validate_jsonpath(jsonpath)

    # Try each handler in sequence, returning the first successful result

    # Handle nested arrays with wildcards or predicates specially
    expr = handle_nested_arrays_special(path)
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
