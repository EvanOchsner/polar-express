import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import polars as pl
from jsonpath_to_polars import jsonpath_to_polars


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
        expected = pl.col("relationships").str.json_decode(
            pl.List(pl.Struct([pl.Field("dest", pl.String)]))
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

        # Create the expected expression manually using the JSONPath approach
        expected = pl.col("items").str.json_path_match("$[?(@.price>10)].name")

        # Use meta.eq for comparison
        assert result.meta.eq(expected)

    def test_multiple_array_indices(self):
        """Test multiple array indices."""
        result = jsonpath_to_polars("$.matrix[0][1]")
        expected = pl.col("matrix").str.json_path_match("$[0][1]")
        assert result.meta.eq(expected)

    def test_wildcard_with_nested_field(self):
        """Test wildcard with nested field."""
        result = jsonpath_to_polars("$.users[*].address.city")
        expected = pl.col("users").str.json_decode(
            pl.List(
                pl.Struct(
                    [pl.Field("address", pl.Struct([pl.Field("city", pl.String)]))]
                )
            )
        )
        assert result.meta.eq(expected)

    def test_wildcard_with_deeply_nested_field(self):
        """Test wildcard with deeply nested fields."""
        result = jsonpath_to_polars("$.users[*].contact.address.city")
        expected = pl.col("users").str.json_decode(
            pl.List(
                pl.Struct(
                    [
                        pl.Field(
                            "contact",
                            pl.Struct(
                                [
                                    pl.Field(
                                        "address",
                                        pl.Struct([pl.Field("city", pl.String)]),
                                    )
                                ]
                            ),
                        )
                    ]
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

    def test_array_index_with_nested_arrays(self):
        """Test array index access with nested arrays."""
        result = jsonpath_to_polars("$.schools[0].classes[*].students[*].grade")
        # With the new approach, we return up to the first wildcard as a string
        expected = pl.col("schools").str.json_path_match("$[0].classes").cast(pl.Utf8)
        assert result.meta.eq(expected)

    def test_complex_path_with_column_and_nested_arrays(self):
        """Test complex path with column name, specific array index, and nested wildcards."""
        result = jsonpath_to_polars(
            "$.education_data.schools[0].classes[*].students[*].grade"
        )
        # The function should extract the column name and properly handle the nested path
        expected = (
            pl.col("education_data")
            .str.json_path_match("$.schools[0].classes")
            .cast(pl.Utf8)
        )
        assert result.meta.eq(expected)

    def test_invalid_jsonpath(self):
        """Test invalid JSONPath format."""
        # Missing $ prefix
        with pytest.raises(ValueError):
            jsonpath_to_polars("types")

        # Invalid starting path
        with pytest.raises(ValueError):
            jsonpath_to_polars("$[0]")
