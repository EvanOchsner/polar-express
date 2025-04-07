"""Tests for the TableMapper class."""

import json

import polars as pl

from polar_express.core.jsonpath_expr import JSONPathExpr
from polar_express.core.table_mapper import TableMapper


class TestTableMapper:
    """Tests for the TableMapper class."""

    def test_init(self) -> None:
        """Test that the TableMapper initializes with empty columns."""
        mapper = TableMapper()
        assert mapper.columns == {}

    def test_add_column(self) -> None:
        """Test adding a single column."""
        mapper = TableMapper()
        result = mapper.add_column("name", "$.name")

        # Test method chaining
        assert result is mapper

        # Test that column was added
        assert "name" in mapper.columns
        assert isinstance(mapper.columns["name"], JSONPathExpr)
        assert mapper.columns["name"].jsonpath == "$.name"
        assert mapper.columns["name"].alias == "name"

    def test_add_columns(self) -> None:
        """Test adding multiple columns."""
        mapper = TableMapper()
        column_defs = {"name": "$.name", "age": "$.age", "address": "$.address.street"}

        result = mapper.add_columns(column_defs)

        # Test method chaining
        assert result is mapper

        # Test that columns were added
        assert set(mapper.columns.keys()) == {"name", "age", "address"}
        assert all(isinstance(expr, JSONPathExpr) for expr in mapper.columns.values())
        assert mapper.columns["name"].jsonpath == "$.name"
        assert mapper.columns["age"].jsonpath == "$.age"
        assert mapper.columns["address"].jsonpath == "$.address.street"

    def test_to_mapper(self) -> None:
        """Test conversion to PolarMapper."""
        mapper = TableMapper()
        mapper.add_columns({"name": "$.name", "age": "$.age"})

        polar_mapper = mapper.to_mapper()

        # Check that the PolarMapper has a select step with the right expressions
        assert len(polar_mapper.steps) == 1
        assert polar_mapper.steps[0]["type"] == "select"
        assert len(polar_mapper.steps[0]["exprs"]) == 2

        # Check that expressions are aliased correctly
        expr_strings = [str(expr) for expr in polar_mapper.steps[0]["exprs"]]
        assert any("name" in expr for expr in expr_strings)
        assert any("age" in expr for expr in expr_strings)

    def test_map(self) -> None:
        """Test mapping data from a DataFrame."""
        # Create a test DataFrame with JSON data
        data = [
            {"json": json.dumps({"name": "Alice", "age": 30, "hobbies": ["reading", "hiking"]})},
            {"json": json.dumps({"name": "Bob", "age": 25, "hobbies": ["gaming", "cooking"]})},
        ]
        df = pl.DataFrame(data)

        # Create a TableMapper and apply it
        mapper = TableMapper()
        mapper.add_columns({"name": "$.json.name", "age": "$.json.age", "first_hobby": "$.json.hobbies[0]"})

        result_df = mapper.map(df)

        # Check the result
        assert result_df.columns == ["name", "age", "first_hobby"]
        assert result_df.shape == (2, 3)
        assert result_df["name"].to_list() == ["Alice", "Bob"]
        assert result_df["age"].to_list() == [str(30), str(25)]
        assert result_df["first_hobby"].to_list() == ["reading", "gaming"]

    def test_get_column_expr(self) -> None:
        """Test getting a column expression by name."""
        mapper = TableMapper()
        mapper.add_column("name", "$.name")

        expr = mapper.get_column_expr("name")
        assert expr is not None
        assert expr.jsonpath == "$.name"

        # Test getting a non-existent column
        assert mapper.get_column_expr("nonexistent") is None

    def test_get_schema(self) -> None:
        """Test getting the schema as a dictionary."""
        mapper = TableMapper()
        schema = {
            "name": "$.name",
            "age": "$.age",
        }
        mapper.add_columns(schema)

        result_schema = mapper.get_schema()
        assert result_schema == schema

    def test_from_dict(self) -> None:
        """Test creating a TableMapper from a dictionary."""
        mapper = TableMapper()
        schema = {
            "name": "$.name",
            "age": "$.age",
        }

        result = mapper.from_dict(schema)

        # Test method chaining
        assert result is mapper

        # Test that columns were added
        assert set(mapper.columns.keys()) == {"name", "age"}
        assert mapper.columns["name"].jsonpath == "$.name"
        assert mapper.columns["age"].jsonpath == "$.age"

    def test_from_schema_dict_class_method(self) -> None:
        """Test the class method for creating a TableMapper from a schema dict."""
        schema = {
            "name": "$.name",
            "age": "$.age",
        }

        mapper = TableMapper.from_schema_dict(schema)

        assert isinstance(mapper, TableMapper)
        assert set(mapper.columns.keys()) == {"name", "age"}
        assert mapper.columns["name"].jsonpath == "$.name"
        assert mapper.columns["age"].jsonpath == "$.age"

    def test_str_repr(self) -> None:
        """Test string representation."""
        mapper = TableMapper()
        mapper.add_columns(
            {
                "name": "$.name",
                "age": "$.age",
            }
        )

        str_repr = str(mapper)
        assert "TableMapper with columns:" in str_repr
        assert "name: $.name" in str_repr
        assert "age: $.age" in str_repr

        # Test empty mapper
        empty_mapper = TableMapper()
        assert str(empty_mapper) == "TableMapper with no columns defined"

        # Test repr
        assert "TableMapper(columns=2)" in repr(mapper)
