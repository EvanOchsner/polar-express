"""Token handling utilities for JSONPath parsing."""

from typing import Any, Dict, List, Optional, Tuple, Union

from polar_express.parsing import predicate_parser

# Type for tokens produced during parsing
Token = Tuple[str, Optional[Union[str, int, Dict[str, Any]]]]


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
    fields = predicate_parser.extract_fields_from_predicate(predicate_str)
    pred_expr = predicate_parser.convert_to_polars(predicate_str)
    return ("predicate", {"expr": pred_expr, "fields": fields})
