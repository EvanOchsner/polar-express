import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import pytest

from jsonpath_to_polars import (
    comparison_to_expr,
    jsonpath_to_polars,
    parse_predicate_tokens,
)


class TestJsonPathToPolars:
    """Test JSONPath to Polars expression conversion."""

    def test_simple_field_access(self):
        """Test simple field access."""
        result = jsonpath_to_polars("$.types")
        expected = pl.col("types")
        assert result.meta.eq(expected)

    def test_nested_field_access(self):
        """Test nested field access."""
        result = jsonpath_to_polars("$.configuration.details.ip")
        expected = pl.col("configuration").str.json_path_match("$.details.ip")
        assert result.meta.eq(expected)

    def test_array_wildcard_access(self):
        """Test array wildcard access."""
        result = jsonpath_to_polars("$.relationships[*].dest")

        # Create the expected expression with empty list handling
        expected = (
            pl.when(
                # Check if it's an empty list or null
                pl.col("relationships")
                .eq("[]")
                .or_(pl.col("relationships").is_null())
            )
            .then(
                # Return null for empty lists
                pl.lit(None)
            )
            .otherwise(
                # Only try to decode when it's not empty
                pl.col("relationships").str.json_decode(
                    pl.List(pl.Struct([pl.Field("dest", pl.String)]))
                )
            )
        )

        assert result.meta.eq(expected)

    def test_array_index_access(self):
        """Test array index access."""
        result = jsonpath_to_polars("$.foo[0]")
        expected = pl.col("foo").str.json_path_match("$[0]")
        assert result.meta.eq(expected)

    def test_array_negative_index_with_nested_field(self):
        """Test array negative index with nested field."""
        result = jsonpath_to_polars("$.foo[-1].bar")
        expected = pl.col("foo").str.json_path_match("$[-1].bar")
        assert result.meta.eq(expected)

    def test_array_with_predicate(self):
        """Test array with predicate filter."""
        result = jsonpath_to_polars("$.items[?(@.price>10)].name")

        # Create the expected expression with empty list handling
        expected = (
            pl.when(
                # Check if it's an empty list or null
                pl.col("items")
                .eq("[]")
                .or_(pl.col("items").is_null())
            )
            .then(
                # Return null for empty lists
                pl.lit(None)
            )
            .otherwise(
                # Only try to decode when it's not empty
                pl.col("items").str.json_decode()
            )
            .filter(
                # Apply the filter condition
                pl.col("price")
                > float(10)
            )
            .list.eval(pl.element().struct.field("name"))
        )

        # polars equality check seems to be broken?? Comparing string representations instead
        assert str(result) == str(expected)
        # assert result.meta.eq(expected)

    def test_array_with_complex_predicate(self):
        """Test array with complex predicate using AND and OR operators."""
        result = jsonpath_to_polars(
            "$.products[?(@.price>10 && @.stock>0 || @.featured==true)].name"
        )

        # Create the expected expression with empty list handling
        expected = (
            pl.when(
                # Check if it's an empty list or null
                pl.col("products")
                .eq("[]")
                .or_(pl.col("products").is_null())
            )
            .then(
                # Return null for empty lists
                pl.lit(None)
            )
            .otherwise(
                # Only try to decode when it's not empty
                pl.col("products").str.json_decode()
            )
            .filter(
                # Apply the complex filter condition with AND/OR logic
                (pl.col("price") > float(10))
                .and_(pl.col("stock") > float(0))
                .or_(pl.col("featured") == True)
            )
            .list.eval(pl.element().struct.field("name"))
        )

        # polars equality check seems to be broken?? Comparing string representations instead
        assert str(result) == str(expected)
        # assert result.meta.eq(expected)

    def test_wildcard_with_nested_field(self):
        """Test wildcard with nested field."""
        result = jsonpath_to_polars("$.users[*].address.city")
        expected = (
            pl.when(pl.col("users").eq("[]").or_(pl.col("users").is_null()))
            .then(pl.lit(None))
            .otherwise(
                pl.col("users").str.json_decode(
                    pl.List(
                        pl.Struct(
                            [
                                pl.Field(
                                    "address", pl.Struct([pl.Field("city", pl.String)])
                                )
                            ]
                        )
                    )
                )
            )
        )
        assert result.meta.eq(expected)

    def test_wildcard_with_deeply_nested_field(self):
        """Test wildcard with deeply nested fields."""
        result = jsonpath_to_polars("$.users[*].contact.address.city")
        expected = (
            pl.when(pl.col("users").eq("[]").or_(pl.col("users").is_null()))
            .then(pl.lit(None))
            .otherwise(
                pl.col("users").str.json_decode(
                    pl.List(
                        pl.Struct(
                            [
                                pl.Field(
                                    "contact",
                                    pl.Struct(
                                        [
                                            pl.Field(
                                                "address",
                                                pl.Struct(
                                                    [pl.Field("city", pl.String)]
                                                ),
                                            )
                                        ]
                                    ),
                                )
                            ]
                        )
                    )
                )
            )
        )
        assert result.meta.eq(expected)

    def test_multiple_arrays_with_wildcards(self):
        """Test path with multiple array wildcards."""
        result = jsonpath_to_polars("$.departments[*].employees[*].name")
        # With the new approach, we return the departments array as a string
        expected = pl.col("departments").cast(pl.Utf8)
        assert result.meta.eq(expected)

    def test_invalid_jsonpath(self):
        """Test invalid JSONPath."""
        # Missing $ prefix
        with pytest.raises(ValueError):
            jsonpath_to_polars("types")

        # Invalid starting path
        with pytest.raises(ValueError):
            jsonpath_to_polars("$[0]")

    def test_parse_predicate_tokens(self):
        """Test parsing predicate tokens into comparison and combinator tokens."""
        # Simple predicate
        tokens = parse_predicate_tokens("@.price > 10")
        assert len(tokens) == 1
        assert tokens[0].meta.eq(comparison_to_expr("price", ">", 10))

        # Predicate with AND combinator
        tokens = parse_predicate_tokens("@.price > 10 && @.stock > 0")
        assert len(tokens) == 3
        assert tokens[0].meta.eq(comparison_to_expr("price", ">", 10))
        assert tokens[1] == "&&"
        assert tokens[2].meta.eq(comparison_to_expr("stock", ">", 0))

        # Predicate with OR combinator
        tokens = parse_predicate_tokens("@.price > 10 || @.featured == true")
        assert len(tokens) == 3
        assert tokens[0].meta.eq(comparison_to_expr("price", ">", 10))
        assert tokens[1] == "||"
        assert tokens[2].meta.eq(comparison_to_expr("featured", "==", True))

        # Predicate with multiple combinators (AND and OR)
        tokens = parse_predicate_tokens(
            "@.price > 10 && @.stock > 0 || @.featured == true"
        )
        assert len(tokens) == 5
        assert tokens[0].meta.eq(comparison_to_expr("price", ">", 10))
        assert tokens[1] == "&&"
        assert tokens[2].meta.eq(comparison_to_expr("stock", ">", 0))
        assert tokens[3] == "||"
        assert tokens[4].meta.eq(comparison_to_expr("featured", "==", True))

        # Test with string value
        tokens = parse_predicate_tokens('@.category == "electronics"')
        assert len(tokens) == 1
        assert tokens[0].meta.eq(comparison_to_expr("category", "==", "electronics"))
