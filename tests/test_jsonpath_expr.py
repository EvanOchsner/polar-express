"""Unit tests for the JSONPathExpr class."""

import pytest
import polars as pl
from jsonpath_expr import JSONPathExpr


class TestJSONPathExpr:
    """Comprehensive tests for the JSONPathExpr class."""

    # Collection of all JSONPath test cases
    jsonpath_cases = [
        "$.name",                    # Simple field access
        "$.users[0].email",          # Array index with field
        "$.items[*].price",          # Wildcard array
        "$.users[?(@.age>30)].name"  # Predicate filter
    ]

    @pytest.mark.parametrize("path", jsonpath_cases)
    def test_init_without_alias(self, path):
        """Test initialization without an alias."""
        expr = JSONPathExpr(path)
        
        # Basic property checks
        assert expr.jsonpath == path
        assert expr.alias is None
        assert isinstance(expr.expr, pl.Expr)

    @pytest.mark.parametrize("path", jsonpath_cases)
    def test_init_with_alias(self, path):
        """Test initialization with an explicit alias."""
        alias = "test_alias"
        expr = JSONPathExpr(path, alias=alias)
        
        # Verify alias is correctly set
        assert expr.jsonpath == path
        assert expr.alias == alias
        assert isinstance(expr.expr, pl.Expr)
        assert expr.expr_str().endswith(f'.alias("{alias}")')

    @pytest.mark.parametrize("path", jsonpath_cases)
    def test_init_with_auto_alias(self, path):
        """Test initialization with an automatically generated alias."""
        # Create alias from the last segment of the path
        auto_alias = f"{path.split('.')[-1]}_field"
        expr = JSONPathExpr(path, alias=auto_alias)
        
        # Verify the path and alias were stored correctly
        assert expr.jsonpath == path
        assert expr.alias == auto_alias
        assert isinstance(expr.expr, pl.Expr)
        assert expr.expr_str().endswith(f'.alias("{auto_alias}")')

    @pytest.mark.parametrize("path", jsonpath_cases)
    def test_string_representations(self, path):
        """Test the string representation methods."""
        expr = JSONPathExpr(path)
        alias = "test_alias"
        expr_with_alias = JSONPathExpr(path, alias=alias)
        
        # Test __str__
        str_repr = str(expr)
        assert f"JSONPath: {path}" in str_repr
        assert "Polars: " in str_repr
        
        # Test __repr__
        assert repr(expr) == f"JSONPathExpr('{path}')"
        assert repr(expr_with_alias) == f"JSONPathExpr('{path}', alias='{alias}')"
        
        # Test jsonpath_str
        assert expr.jsonpath_str() == path
        
        # Test expr_str
        assert expr.expr_str() == str(expr.expr)

    @pytest.mark.parametrize("path", jsonpath_cases)
    def test_tree_diagram(self, path):
        """Test the tree_diagram method."""
        expr = JSONPathExpr(path)
        
        # Get the tree diagram
        tree = expr.tree_diagram()
        
        # Verify it's a non-empty string
        assert isinstance(tree, str)
        assert len(tree) > 0
        
        # Verify it contains expected tree structure indicators
        assert any(char in tree for char in ["│", "┌", "┐", "─", "└", "┘"])

    @pytest.mark.parametrize("path", jsonpath_cases)
    def test_with_alias(self, path):
        """Test that with_alias creates a new instance with updated alias."""
        # Original object
        original_alias = "original_alias"
        expr = JSONPathExpr(path, alias=original_alias)
        
        # New object with different alias
        new_alias = "new_alias"
        new_expr = expr.with_alias(new_alias)
        
        # Verify it's a different instance
        assert new_expr is not expr
        
        # Verify path stays the same
        assert new_expr.jsonpath == expr.jsonpath
        
        # Verify new alias is applied
        assert new_expr.alias == new_alias
        assert new_expr.expr_str().endswith(f'.alias("{new_alias}")')
        
        # Verify original object is unchanged
        assert expr.alias == original_alias
        assert expr.expr_str().endswith(f'.alias("{original_alias}")')

    def test_expected_expressions_simple_field(self):
        """Test the generated expression for a simple field access."""
        path = "$.name"
        expr = JSONPathExpr(path)
        expr_str = expr.expr_str()
        
        assert "col(\"name\")" in expr_str

    def test_expected_expressions_array_index(self):
        """Test the generated expression for array index access."""
        path = "$.users[0].email"
        expr = JSONPathExpr(path)
        expr_str = expr.expr_str()
        
        assert "col(\"users\")" in expr_str
        assert "json_path_match" in expr_str
        assert "$[0].email" in expr_str

    def test_expected_expressions_wildcard_array(self):
        """Test the generated expression for wildcard array access."""
        path = "$.items[*].price"
        expr = JSONPathExpr(path)
        expr_str = expr.expr_str()
        
        assert "col(\"items\")" in expr_str
        assert "json_decode" in expr_str or "json_path_match" in expr_str

    def test_expected_expressions_predicate_filter(self):
        """Test the generated expression for predicate filter."""
        path = "$.users[?(@.age>30)].name"
        expr = JSONPathExpr(path)
        expr_str = expr.expr_str()
        
        assert "col(\"users\")" in expr_str
        # Additional assertions depending on how predicates are handled