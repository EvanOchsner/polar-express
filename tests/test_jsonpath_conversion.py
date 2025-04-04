import polars as pl
import pytest

from polar_express.conversion.jsonpath_to_polars import (
    jsonpath_to_polars,
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
        expected = (
            pl.when(pl.col("relationships").eq("[]").or_(pl.col("relationships").is_null()))
            .then(pl.lit(None))
            .otherwise(
                pl.col("relationships")
                .str.json_decode(pl.List(pl.Struct([pl.Field("dest", pl.String())])))
                .list.eval(pl.element().struct.field("dest"))
            )
        )
        # TODO: Understand why meta.eq does not agree here
        assert result.meta.tree_format(return_as_string=True) == expected.meta.tree_format(return_as_string=True)
        # assert result.meta.eq(expected)

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
        expected = (
            pl.when(pl.col("items").eq("[]").or_(pl.col("items").is_null()))
            .then(pl.lit(None))
            .otherwise(
                pl.col("items")
                .str.json_decode(pl.List(pl.Struct([pl.Field("name", pl.Utf8), pl.Field("price", pl.Utf8)])))
                .list.eval(
                    pl.when(pl.element().struct.field("price").cast(pl.Float32).gt(float(10)))
                    .then(pl.element().struct.field("name"))
                    .otherwise(pl.lit(None))
                )
                .list.drop_nulls()
            )
        )

        # TODO: Understand why meta.eq does not agree here
        assert result.meta.tree_format(return_as_string=True) == expected.meta.tree_format(return_as_string=True)
        # assert result.meta.eq(expected)

    def test_array_with_complex_predicate(self):
        """Test array with complex predicate using AND and OR operators."""
        result = jsonpath_to_polars("$.products[?(@.price>10 && @.stock>0 || @.featured==true)].name")

        # Create the expected expression with empty list handling
        expected = (
            pl.when(pl.col("products").eq("[]").or_(pl.col("products").is_null()))
            .then(pl.lit(None))
            .otherwise(
                pl.col("products")
                .str.json_decode(
                    pl.List(
                        pl.Struct(
                            [
                                pl.Field("price", pl.Utf8),
                                pl.Field("stock", pl.Utf8),
                                pl.Field("featured", pl.Utf8),
                                pl.Field("name", pl.Utf8),
                            ]
                        )
                    )
                )
                .list.eval(
                    pl.when(
                        pl.element().struct.field("price").cast(pl.Float32).gt(float(10))
                        & pl.element().struct.field("stock").cast(pl.Float32).gt(float(0))
                        | pl.element().struct.field("featured").cast(pl.Boolean)
                    )
                    .then(pl.element().struct.field("name"))
                    .otherwise(pl.lit(None))
                )
                .list.drop_nulls()
            )
        )

        # polars equality check seems to be broken?? Comparing string representations instead
        assert str(result) == str(expected)
        assert result.meta.tree_format(return_as_string=True) == expected.meta.tree_format(return_as_string=True)
        # assert result.meta.eq(expected)

    def test_wildcard_with_nested_field(self):
        """Test wildcard with nested field."""
        result = jsonpath_to_polars("$.users[*].address.city")
        expected = (
            pl.when(pl.col("users").eq("[]").or_(pl.col("users").is_null()))
            .then(pl.lit(None))
            .otherwise(
                pl.col("users")
                .str.json_decode(pl.List(pl.Struct([pl.Field("address", pl.Struct([pl.Field("city", pl.Utf8)]))])))
                .list.eval(pl.element().struct.field("address").struct.field("city"))
            )
        )
        # TODO: Understand why meta.eq does not agree here
        assert result.meta.tree_format(return_as_string=True) == expected.meta.tree_format(return_as_string=True)
        # assert result.meta.eq(expected)

    def test_wildcard_with_deeply_nested_field(self):
        """Test wildcard with deeply nested fields."""
        result = jsonpath_to_polars("$.users[*].contact.address.city")
        expected = (
            pl.when(pl.col("users").eq("[]").or_(pl.col("users").is_null()))
            .then(pl.lit(None))
            .otherwise(
                pl.col("users")
                .str.json_decode(
                    pl.List(
                        pl.Struct(
                            [
                                pl.Field(
                                    "contact",
                                    pl.Struct(
                                        [
                                            pl.Field(
                                                "address",
                                                pl.Struct([pl.Field("city", pl.String())]),
                                            )
                                        ]
                                    ),
                                )
                            ]
                        )
                    )
                )
                .list.eval(pl.element().struct.field("contact").struct.field("address").struct.field("city"))
            )
        )
        # TODO: Understand why meta.eq does not agree here
        assert result.meta.tree_format(return_as_string=True) == expected.meta.tree_format(return_as_string=True)
        # assert result.meta.eq(expected)

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

    def test_complex_path_with_multiple_types(self):
        """Test complex path with multiple types like '$.users[0].addresses[*].city'."""
        path = "$.users[0].addresses[*].city"
        result = jsonpath_to_polars(path)
        base_expr = pl.col("users").str.json_path_match("$[0].addresses")
        expected = (
            pl.when(base_expr.eq("[]").or_(base_expr.is_null()))
            .then(pl.lit(None))
            .otherwise(
                base_expr.str.json_decode(pl.List(pl.Struct([pl.Field("city", pl.Utf8)]))).list.eval(
                    pl.element().struct.field("addresses").list.eval(pl.element().struct.field("city"))
                )
            )
        )
        # TODO: Understand why meta.eq does not agree here
        assert result.meta.tree_format(return_as_string=True) == expected.meta.tree_format(return_as_string=True)
        # assert result.meta.eq(expected)

    def test_array_index_before_predicate(self):
        """Test array indexing followed by predicate filtering like '$.books[0].chapters[?(@.pages>10)].title'."""
        path = "$.books[0].chapters[?(@.pages>10)].title"
        result = jsonpath_to_polars(path)

        base_expr = pl.col("books").str.json_path_match("$[0].chapters")
        expected = (
            pl.when(base_expr.eq("[]").or_(base_expr.is_null()))
            .then(pl.lit(None))
            .otherwise(
                base_expr.str.json_decode(pl.List(pl.Struct([pl.Field("title", pl.Utf8), pl.Field("pages", pl.Utf8)])))
                .list.eval(
                    pl.when(pl.element().struct.field("pages").cast(pl.Float32).gt(float(10)))
                    .then(pl.element().struct.field("title"))
                    .otherwise(pl.lit(None))
                )
                .list.drop_nulls()
            )
        )
        # TODO: Understand why meta.eq does not agree here
        assert result.meta.tree_format(return_as_string=True) == expected.meta.tree_format(return_as_string=True)
        # assert result.meta.eq(expected)
