"""TableMapper class for defining table schemas using JSONPath expressions."""

from typing import Dict, Optional

from polars import DataFrame

from polar_express.core.jsonpath_expr import JSONPathExpr
from polar_express.core.polar_mapper import PolarMapper


class TableMapper:
    """
    A class that defines table columns using JSONPath strings.

    TableMapper allows defining a table schema where each column is specified
    using a JSONPath expression. These expressions are converted to polars
    expressions for data extraction.

    Attributes:
        columns: Dictionary mapping column names to their JSONPathExpr objects
    """

    def __init__(self) -> None:
        """Initialize an empty TableMapper with no columns defined."""
        self.columns: Dict[str, JSONPathExpr] = {}

    def add_column(self, name: str, jsonpath: str) -> "TableMapper":
        """
        Add a column definition to the table schema.

        Args:
            name: The name of the column in the resulting table
            jsonpath: The JSONPath expression to extract the column data

        Returns:
            Self for method chaining
        """
        self.columns[name] = JSONPathExpr(jsonpath, name)
        return self

    def add_columns(self, column_defs: Dict[str, str]) -> "TableMapper":
        """
        Add multiple column definitions to the table schema.

        Args:
            column_defs: Dictionary mapping column names to JSONPath expressions

        Returns:
            Self for method chaining
        """
        for name, jsonpath in column_defs.items():
            self.add_column(name, jsonpath)
        return self

    def to_mapper(self) -> PolarMapper:
        """
        Convert the table schema to a PolarMapper for data extraction.

        Returns:
            A PolarMapper configured to extract the defined columns
        """
        mapper = PolarMapper()
        exprs = [col_expr.expr for col_expr in self.columns.values()]
        mapper.add_select_step(exprs)
        return mapper

    def map(self, df: DataFrame) -> DataFrame:
        """
        Apply the table schema to extract data from the input DataFrame.

        Args:
            df: Input DataFrame with JSON/nested data

        Returns:
            A DataFrame with columns extracted according to the schema
        """
        return self.to_mapper().map(df)

    def get_column_expr(self, name: str) -> Optional[JSONPathExpr]:
        """
        Get the JSONPathExpr for a specific column.

        Args:
            name: The name of the column

        Returns:
            The JSONPathExpr for the column or None if not found
        """
        return self.columns.get(name)

    def get_schema(self) -> Dict[str, str]:
        """
        Get the schema as a dictionary of column names to JSONPath expressions.

        Returns:
            Dictionary mapping column names to their JSONPath expressions
        """
        return {name: expr.jsonpath for name, expr in self.columns.items()}

    def from_dict(self, schema_dict: Dict[str, str]) -> "TableMapper":
        """
        Create a TableMapper from a dictionary schema definition.

        Args:
            schema_dict: Dictionary mapping column names to JSONPath expressions

        Returns:
            Self with columns defined from the schema dictionary
        """
        self.columns = {}
        return self.add_columns(schema_dict)

    @classmethod
    def from_schema_dict(cls, schema_dict: Dict[str, str]) -> "TableMapper":
        """
        Create a new TableMapper from a dictionary schema definition.

        Args:
            schema_dict: Dictionary mapping column names to JSONPath expressions

        Returns:
            A new TableMapper with columns defined from the schema dictionary
        """
        mapper = cls()
        return mapper.add_columns(schema_dict)

    def __str__(self) -> str:
        """
        Return a string representation of the TableMapper.

        Returns:
            A string showing the table schema with column names and JSONPath expressions
        """
        if not self.columns:
            return "TableMapper with no columns defined"

        result = "TableMapper with columns:\n"
        for name, expr in self.columns.items():
            result += f"- {name}: {expr.jsonpath}\n"
        return result

    def __repr__(self) -> str:
        """
        Return a representation of the TableMapper.

        Returns:
            A string representation of the TableMapper
        """
        return f"TableMapper(columns={len(self.columns)})"
