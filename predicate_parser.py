"""
Parser for converting logical statements into Boolean syntax trees (BSTs) and BSTs into polars expressions.

This module provides functionality to parse logical expressions with AND, OR operators,
parentheses, and conditions into a tree structure.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

import polars as pl


class NodeType(Enum):
    """Types of nodes in the Boolean syntax tree."""

    AND = "AND"
    OR = "OR"
    CONDITION = "CONDITION"


class Node:
    """Represents a node in the Boolean syntax tree."""

    def __init__(self, node_type: NodeType, value: Optional[str] = None):
        self.type = node_type
        self.value = value
        self.children: List[Node] = []

    def __str__(self) -> str:
        """String representation of the node."""
        if self.type == NodeType.CONDITION:
            return f"Condition({self.value})"
        elif self.type == NodeType.AND:
            return f"AND({', '.join(str(child) for child in self.children)})"
        elif self.type == NodeType.OR:
            return f"OR({', '.join(str(child) for child in self.children)})"
        return ""

    def to_dict(self) -> Mapping[str, Any]:
        """Convert the node to a dictionary representation."""
        result: Dict[str, Any] = {"type": self.type.value}

        if self.type == NodeType.CONDITION:
            if self.value is not None:
                result["value"] = self.value
        else:
            result["children"] = [child.to_dict() for child in self.children]

        return result

    def to_ascii(self) -> str:
        """Generate an ASCII representation of the tree with this node as root."""
        lines: List[str] = []
        self._to_ascii_rec(lines, "", "", "")
        return "\n".join(lines)

    def _to_ascii_rec(self, lines: List[str], prefix: str, child_prefix: str, label_prefix: str) -> None:
        """Recursively build ASCII representation of the tree."""
        label = str(self.type.value)
        if self.type == NodeType.CONDITION:
            label += f": {self.value}"

        lines.append(f"{prefix}{label_prefix}{label}")

        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1

            if is_last_child:
                new_prefix = child_prefix + "    "
                new_label_prefix = "└── "
            else:
                new_prefix = child_prefix + "│   "
                new_label_prefix = "├── "

            child._to_ascii_rec(lines, child_prefix, new_prefix, new_label_prefix)


class TokenType(Enum):
    """Types of tokens in the logical expression."""

    AND = "&&"
    OR = "||"
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    CONDITION = "CONDITION"


class Token:
    """Represents a token in the logical expression."""

    def __init__(self, token_type: TokenType, value: Optional[str] = None):
        self.type = token_type
        self.value = value

    def __str__(self) -> str:
        """String representation of the token."""
        if self.type == TokenType.CONDITION:
            return f"Condition({self.value})"
        return str(self.type.value)


def tokenize(expression: str) -> List[Token]:
    """
    Tokenize a logical expression into a list of tokens.

    Args:
        expression: The logical expression to tokenize

    Returns:
        A list of tokens

    Raises:
        ValueError: If the expression is invalid or contains unbalanced quotes
    """
    tokens = []
    i = 0
    n = len(expression)

    while i < n:
        # Skip whitespace
        if expression[i].isspace():
            i += 1
            continue

        # Check for operators and parentheses
        if i + 1 < n and expression[i : i + 2] == "&&":
            tokens.append(Token(TokenType.AND))
            i += 2
        elif i + 1 < n and expression[i : i + 2] == "||":
            tokens.append(Token(TokenType.OR))
            i += 2
        elif expression[i] == "(":
            tokens.append(Token(TokenType.LEFT_PAREN))
            i += 1
        elif expression[i] == ")":
            tokens.append(Token(TokenType.RIGHT_PAREN))
            i += 1
        else:
            # This is a condition
            start = i
            in_quotes = False
            quote_char = None

            # Continue until we hit an operator or a parenthesis (ignoring those in quotes)
            while i < n:
                # Handle quoted strings
                if expression[i] in ('"', "'"):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = expression[i]
                    elif expression[i] == quote_char:
                        in_quotes = False
                        quote_char = None

                # Check for operators and parentheses (outside quotes)
                if not in_quotes:
                    if i + 1 < n and (expression[i : i + 2] == "&&" or expression[i : i + 2] == "||"):
                        break
                    if expression[i] == "(" or expression[i] == ")":
                        break

                i += 1

            # Check for unbalanced quotes
            if in_quotes:
                raise ValueError(f"Unbalanced quotes in condition starting at position {start}")

            condition = expression[start:i].strip()
            if condition:
                tokens.append(Token(TokenType.CONDITION, condition))

    return tokens


def parse_expression(tokens: List[Token]) -> Node:
    """
    Parse a list of tokens into a Boolean syntax tree.

    Args:
        tokens: The list of tokens to parse

    Returns:
        The root node of the Boolean syntax tree

    Raises:
        ValueError: If the tokens form an invalid expression
    """

    def parse_or_expr(index: int) -> Tuple[Node, int]:
        """Parse an OR expression."""
        left, index = parse_and_expr(index)

        while index < len(tokens) and tokens[index].type == TokenType.OR:
            # Create OR node
            or_node = Node(NodeType.OR)
            or_node.children.append(left)

            # Parse right operand
            index += 1
            right, index = parse_and_expr(index)
            or_node.children.append(right)

            left = or_node

        return left, index

    def parse_and_expr(index: int) -> Tuple[Node, int]:
        """Parse an AND expression."""
        left, index = parse_primary(index)

        while index < len(tokens) and tokens[index].type == TokenType.AND:
            # Create AND node
            and_node = Node(NodeType.AND)
            and_node.children.append(left)

            # Parse right operand
            index += 1
            right, index = parse_primary(index)
            and_node.children.append(right)

            left = and_node

        return left, index

    def parse_primary(index: int) -> Tuple[Node, int]:
        """Parse a primary expression (condition or parenthesized expression)."""
        if index >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[index]

        if token.type == TokenType.LEFT_PAREN:
            # Parse expression within parentheses
            index += 1
            expr, index = parse_or_expr(index)

            # Expect closing parenthesis
            if index < len(tokens) and tokens[index].type == TokenType.RIGHT_PAREN:
                index += 1
                return expr, index
            else:
                raise ValueError("Expected closing parenthesis")

        elif token.type == TokenType.CONDITION:
            # Create condition node
            node = Node(NodeType.CONDITION, token.value)
            return node, index + 1

        else:
            raise ValueError(f"Unexpected token: {token.type}")

    if not tokens:
        raise ValueError("Empty expression")

    # Start parsing from the first token
    result, index = parse_or_expr(0)

    # Ensure all tokens were consumed
    if index != len(tokens):
        raise ValueError(f"Unexpected token at position {index}")

    return result


def build_boolean_syntax_tree(expression: str) -> Node:
    """
    Convert a logical statement into a Boolean syntax tree.

    The statement can include:
    - && (logical AND)
    - || (logical OR)
    - ( and ) (parentheses for grouping)
    - Conditions (e.g., @.foo == "value")

    Operator precedence: AND > OR

    Examples:
        >>> tree = build_boolean_syntax_tree('@.key1 == "value1" && @.key2 != "value2"')
        >>> tree.type
        NodeType.AND
        >>> [child.value for child in tree.children]
        ['@.key1 == "value1"', '@.key2 != "value2"']

        >>> tree = build_boolean_syntax_tree('@.k1 == "v1" || (@.k2 == "v2" && (@.k3 >= 1.1 || @.k4 < 0))')
        >>> tree.type
        NodeType.OR

    Args:
        expression: The logical statement to parse

    Returns:
        The root node of the Boolean syntax tree

    Raises:
        ValueError: If the expression is invalid
    """
    tokens = tokenize(expression)
    return parse_expression(tokens)


def extract_fields_from_predicate(predicate: str) -> list[str]:
    """Extract field names from a JSONPath predicate string.

    Args:
        predicate: JSONPath predicate string (e.g. '@.foo == "bar" && @.baz == "bang"')

    Returns:
        List of field names referenced in the predicate
    """
    # Match @.field_name pattern, capturing just the field_name part
    return re.findall(r"@\.([a-zA-Z0-9_]+)", predicate)


def to_polars_expr(tree: Node) -> pl.Expr:
    """
    Convert a Boolean syntax tree to a Polars expression.

    This function translates BST nodes into polars expressions that can be executed
    using the polars query engine. It handles field references, comparison operations,
    and logical operations.

    The expressions are designed to work with a list of structs,
    where each struct can be filtered based on the condition.
    Typically used with pl.col("column_name").list.filter(expr).

    Args:
        tree: The root node of the Boolean syntax tree

    Returns:
        A Polars expression object (pl.Expr)

    Raises:
        ValueError: If a condition has an unsupported format or operation
    """
    if tree.type == NodeType.CONDITION and tree.value is not None:
        # Parse condition: expected format is "@.field op value"
        condition = tree.value.strip()

        # Match field pattern (@.field)
        field_match = re.search(r"(@\.\w+)", condition)
        if not field_match:
            raise ValueError(f"Invalid field reference in condition: {condition}")

        field_ref = field_match.group(1)
        field_name = field_ref.replace("@.", "")

        # Create pl.element().struct.field("field") expression for list of structs access
        pl_field = pl.element().struct.field(field_name)

        # Match comparison operator
        if "==" in condition:
            op_parts = condition.split("==", 1)
            value = op_parts[1].strip()
            # Handle quoted strings
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # Check for boolean values
            if value.lower() == "true":
                return pl_field.eq(True)
            elif value.lower() == "false":
                return pl_field.eq(False)
            else:
                return pl_field.eq(value)
        elif "!=" in condition:
            op_parts = condition.split("!=", 1)
            value = op_parts[1].strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # Check for boolean values
            if value.lower() == "true":
                return pl_field.ne(True)
            elif value.lower() == "false":
                return pl_field.ne(False)
            else:
                return pl_field.ne(value)
        elif ">=" in condition:
            op_parts = condition.split(">=", 1)
            value = op_parts[1].strip()
            # Strip quotes if present for numeric values
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # Try to convert to float for numeric comparisons
            try:
                return pl_field.cast(pl.Float32).ge(float(value))
            except ValueError:
                return pl_field.cast(pl.Float32).ge(value)
        elif "<=" in condition:
            op_parts = condition.split("<=", 1)
            value = op_parts[1].strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            try:
                return pl_field.cast(pl.Float32).le(float(value))
            except ValueError:
                return pl_field.cast(pl.Float32).le(value)
        elif ">" in condition:
            op_parts = condition.split(">", 1)
            value = op_parts[1].strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            try:
                return pl_field.cast(pl.Float32).gt(float(value))
            except ValueError:
                return pl_field.cast(pl.Float32).gt(value)
        elif "<" in condition:
            op_parts = condition.split("<", 1)
            value = op_parts[1].strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            try:
                return pl_field.cast(pl.Float32).lt(float(value))
            except ValueError:
                return pl_field.cast(pl.Float32).lt(value)
        else:
            raise ValueError(f"Unsupported comparison operator in condition: {condition}")

    elif tree.type == NodeType.AND:
        # Convert AND node to .and_() method chain
        if not tree.children:
            raise ValueError("AND node has no children")

        children_exprs = [to_polars_expr(child) for child in tree.children]
        if len(children_exprs) == 1:
            return children_exprs[0]

        # Chain all children with .and_()
        result = children_exprs[0]
        for expr in children_exprs[1:]:
            result = result.and_(expr)

        return result

    elif tree.type == NodeType.OR:
        # Convert OR node to .or_() method chain
        if not tree.children:
            raise ValueError("OR node has no children")

        children_exprs = [to_polars_expr(child) for child in tree.children]
        if len(children_exprs) == 1:
            return children_exprs[0]

        # Chain all children with .or_()
        result = children_exprs[0]
        for expr in children_exprs[1:]:
            result = result.or_(expr)

        return result

    raise ValueError(f"Unsupported node type: {tree.type}")


def convert_to_polars(expression: str) -> pl.Expr:
    """
    Convert a logical predicate string directly to a Polars expression.

    This is a convenience function that combines parsing and conversion to Polars.
    The resulting expression is designed to work with a list of structs in Polars,
    where each struct can be filtered based on the condition.

    Typically used in expressions like:
    ```python
    df.with_column(
        pl.col("struct_list_column").map_elements(
            lambda items: [
                item for item in items
                if evaluate_predicate_on_item(polars_expr, item)
            ]
        )
    )
    ```

    The @. syntax in the predicate refers to fields within each struct.

    Args:
        expression: The logical predicate in string form (e.g., "@.price > 500 && @.in_stock == true")

    Returns:
        A Polars expression object (pl.Expr) that can be used to filter struct elements

    Raises:
        ValueError: If the expression is invalid or contains unsupported operations
    """
    tree = build_boolean_syntax_tree(expression)
    return to_polars_expr(tree)
