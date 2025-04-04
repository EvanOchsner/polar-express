"""
Advanced usage examples for the Polar Express library.

This example demonstrates:
1. Using PolarMapper to build an ETL pipeline
2. Complex JSONPath extraction with array filtering
3. Nested field access patterns
"""

import json
import polars as pl
from polar_express import JSONPathExpr, PolarMapper, jsonpath_to_polars

# Sample JSON data
json_data = {
    "products": [
        {
            "id": "p1",
            "name": "Laptop",
            "price": 1200,
            "specs": {"ram": "16GB", "storage": "512GB"},
            "tags": ["electronics", "computing"],
            "reviews": [
                {"user": "user1", "rating": 4, "verified": True},
                {"user": "user2", "rating": 5, "verified": True},
                {"user": "user3", "rating": 3, "verified": False}
            ]
        },
        {
            "id": "p2",
            "name": "Smartphone",
            "price": 800,
            "specs": {"ram": "8GB", "storage": "256GB"},
            "tags": ["electronics", "mobile"],
            "reviews": [
                {"user": "user4", "rating": 5, "verified": True},
                {"user": "user5", "rating": 2, "verified": True}
            ]
        }
    ]
}

# Create DataFrame with JSON data
df = pl.DataFrame({
    "id": ["doc1"],
    "data": [json.dumps(json_data)]
})

# Create a PolarMapper to build an ETL pipeline
mapper = PolarMapper()

# Step 1: Extract product names
product_names = JSONPathExpr("$.data.products[*].name", alias="product_names")
mapper.add_with_columns_step([product_names.expr])

# Step 2: Extract high-rated verified reviews (rating >= 4 and verified = true)
high_rated_reviews = JSONPathExpr(
    '$.data.products[*].reviews[?(@.rating >= 4 && @.verified == true)].user',
    alias="high_rated_verified_reviewers"
)
mapper.add_with_columns_step([high_rated_reviews.expr])

# Step 3: Extract product specs 
specs = JSONPathExpr("$.data.products[*].specs", alias="product_specs")
mapper.add_with_columns_step([specs.expr])

# Step 4: Select only the columns we want
mapper.add_select_step([
    pl.col("id"),
    pl.col("product_names"),
    pl.col("high_rated_verified_reviewers"),
    pl.col("product_specs")
])

# Apply the transformation pipeline
result = mapper.map(df)

print("ETL Pipeline Results:")
print(result)

print("\nMapper description:")
print(mapper.describe())

# Save the mapper to JSON for later use
mapper_json = mapper.to_json()
print("\nMapper JSON representation:")
print(mapper_json)