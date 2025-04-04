"""Polar Express - Convert JSONPath expressions to Polars expressions."""

__version__ = "0.1.0"

# Core components
from polar_express.core.jsonpath_expr import JSONPathExpr
from polar_express.core.polar_mapper import PolarMapper

# Main conversion function
from polar_express.conversion.jsonpath_to_polars import jsonpath_to_polars

__all__ = ["JSONPathExpr", "PolarMapper", "jsonpath_to_polars"]
