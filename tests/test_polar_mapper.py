import io
import json
from typing import List

import polars as pl
import pytest
from polars import Expr

from polar_mapper import PolarMapper


class TestPolarMapper:
    """
    Unit tests for the PolarMapper class.
    """

    def test_empty_mapper(self):
        """Test an empty mapper with no steps."""
        mapper = PolarMapper()
        assert len(mapper.steps) == 0
        assert str(mapper) == "PolarMapper with no steps"
        assert repr(mapper) == "PolarMapper(steps=0)"

    def test_add_select_step(self):
        """Test adding a select step."""
        mapper = PolarMapper()
        mapper.add_select_step([pl.col("a"), pl.col("b")])
        
        assert len(mapper.steps) == 1
        assert mapper.steps[0]["type"] == "select"
        assert len(mapper.steps[0]["exprs"]) == 2
        assert str(mapper.steps[0]["exprs"][0]) == 'col("a")'
        assert str(mapper.steps[0]["exprs"][1]) == 'col("b")'

    def test_add_with_columns_step(self):
        """Test adding a with_columns step."""
        mapper = PolarMapper()
        mapper.add_with_columns_step([pl.col("a") + 1, pl.col("b") * 2])
        
        assert len(mapper.steps) == 1
        assert mapper.steps[0]["type"] == "with_columns"
        assert len(mapper.steps[0]["exprs"]) == 2
        assert str(mapper.steps[0]["exprs"][0]) == '[(col("a")) + (dyn int: 1)]'
        assert str(mapper.steps[0]["exprs"][1]) == '[(col("b")) * (dyn int: 2)]'

    def test_add_filter_step(self):
        """Test adding a filter step."""
        mapper = PolarMapper()
        mapper.add_filter_step(pl.col("a") > 10)
        
        assert len(mapper.steps) == 1
        assert mapper.steps[0]["type"] == "filter"
        assert str(mapper.steps[0]["expr"]) == '[(col("a")) > (dyn int: 10)]'

    def test_method_chaining(self):
        """Test method chaining for building a pipeline."""
        mapper = PolarMapper()
        result = mapper.add_select_step([pl.col("a"), pl.col("b")]) \
                       .add_with_columns_step([pl.col("a") + pl.col("b")]) \
                       .add_filter_step(pl.col("a") > 0)
        
        assert result is mapper  # Should return self for chaining
        assert len(mapper.steps) == 3
        assert mapper.steps[0]["type"] == "select"
        assert mapper.steps[1]["type"] == "with_columns"
        assert mapper.steps[2]["type"] == "filter"

    def test_map_function(self):
        """Test the map function applies steps in the correct order."""
        # Create a test DataFrame
        df = pl.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40],
            "c": [100, 200, 300, 400]
        })
        
        # Create mapper with a pipeline
        mapper = PolarMapper()
        mapper.add_select_step([pl.col("a"), pl.col("b")]) \
              .add_with_columns_step([(pl.col("a") + pl.col("b")).alias("sum")]) \
              .add_filter_step(pl.col("sum") > 20)
        
        # Apply the mapper
        result_df = mapper.map(df)
        
        # Verify the result
        assert result_df.shape == (3, 3)  # 3 rows, 3 columns (a, b, sum)
        assert result_df.columns == ["a", "b", "sum"]
        assert result_df["sum"].to_list() == [22, 33, 44]  # Only sums > 20

    def test_to_string(self):
        """Test the to_string method."""
        mapper = PolarMapper()
        mapper.add_select_step([pl.col("a"), pl.col("b")]) \
              .add_filter_step(pl.col("a") > 10)
        
        expected = (
            "PolarMapper with steps:\n"
            "1. SELECT: col(\"a\"), col(\"b\")\n"
            "2. FILTER: [(col(\"a\")) > (dyn int: 10)]\n"
        )
        assert mapper.to_string() == expected

    def test_str_and_repr(self):
        """Test the __str__ and __repr__ methods."""
        mapper = PolarMapper()
        mapper.add_select_step([pl.col("a")])
        
        # Verify that __str__ calls to_string
        expected_str = "PolarMapper with steps:\n1. SELECT: col(\"a\")\n"
        assert str(mapper) == expected_str
        assert str(mapper) == mapper.to_string()
        
        # Check repr format
        assert repr(mapper) == "PolarMapper(steps=1)"

    def test_to_json(self):
        """Test the to_json method."""
        mapper = PolarMapper()
        mapper.add_select_step([pl.col("a")]) \
              .add_with_columns_step([pl.col("a") * 2])
        
        expected_json = json.dumps({
            "steps": [
                {"type": "select", "exprs": ["col(\"a\")"]},
                {"type": "with_columns", "exprs": ["[(col(\"a\")) * (dyn int: 2)]"]}
            ]
        }, indent=2)
        
        assert mapper.to_json() == expected_json

    def test_describe_without_output(self):
        """Test the describe method when returning a string."""
        mapper = PolarMapper()
        mapper.add_select_step([pl.col("a"), pl.col("b")])
        
        expected_description = (
            "This mapper performs the following operations on the input DataFrame:\n\n"
            "Step 1: Select columns col(\"a\"), col(\"b\")\n\n"
        )
        
        description = mapper.describe()
        assert isinstance(description, str)
        assert description == expected_description

    def test_describe_with_output(self):
        """Test the describe method when writing to an output stream."""
        mapper = PolarMapper()
        mapper.add_filter_step(pl.col("price") > 100)
        
        expected_output = (
            "This mapper performs the following operations on the input DataFrame:\n\n"
            "Step 1: Filter rows where [(col(\"price\")) > (dyn int: 100)]\n\n"
        )
        
        # Use StringIO as a test output stream
        output = io.StringIO()
        result = mapper.describe(output)
        
        # Should return None when writing to output
        assert result is None
        
        # Check the content was written to the output stream
        output_str = output.getvalue()
        assert output_str == expected_output

    def test_describe_empty_mapper(self):
        """Test the describe method with an empty mapper."""
        mapper = PolarMapper()
        
        description = mapper.describe()
        assert "This mapper performs no operations on the input DataFrame." == description


if __name__ == "__main__":
    pytest.main()