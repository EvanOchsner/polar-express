"""Token handling utilities for JSONPath parsing."""

from typing import Any, Dict, List, Optional, Tuple, Union

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
                        # We don't support non-integer indices
                        raise ValueError(f"Non-integer array index not supported: '{index_str}'")

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
        A predicate token with the predicate string as its value.
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

    # Remove the outer parentheses if present
    if predicate_str.startswith("(") and predicate_str.endswith(")"):
        predicate_str = predicate_str[1:-1]

    # Return just the predicate string without the parentheses
    return ("predicate", predicate_str)


def tokens_to_jsonpath(tokens: List[Token]) -> str:
    """
    Convert a list of tokens back to a JSONPath string.

    Args:
        tokens: The list of tokens to convert.

    Returns:
        A JSONPath string representation of the tokens.
    """
    path = "$"
    for token_type, token_value in tokens:
        if token_type == "field":
            path += f".{token_value}"
        elif token_type == "index":
            path += f"[{token_value}]"
        elif token_type == "wildcard":
            path += "[*]"
        elif token_type == "predicate":
            # token_value now contains the predicate string directly
            path += f"[?({token_value})]"
    return path
