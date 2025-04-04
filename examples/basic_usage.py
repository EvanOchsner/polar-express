"""
Basic usage examples for the Polar Express library.

This example demonstrates simple JSONPath extraction from JSON data stored in a Polars DataFrame.
"""

import polars as pl
from polar_express import JSONPathExpr, jsonpath_to_polars

# Create a DataFrame with JSON data in string columns
df = pl.DataFrame({
    "user_data": ['{"name": "Alice", "address": {"city": "New York"}}']
})

# Method 1: Using jsonpath_to_polars function directly
city_expr = jsonpath_to_polars("$.user_data.address.city")
result1 = df.with_columns([
    city_expr.alias("city")
])

# Method 2: Using JSONPathExpr class for better reusability
city_path = JSONPathExpr("$.user_data.address.city", alias="city")
result2 = df.with_columns([
    city_path.expr
])

print("Using direct function:")
print(result1)

print("\nUsing JSONPathExpr class:")
print(result2)

print("\nExamining the expression:")
print(f"Original JSONPath: {city_path.jsonpath_str()}")
print(f"Polars Expression: {city_path.expr_str()}")