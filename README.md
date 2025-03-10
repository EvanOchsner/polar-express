# Polar Express

A Python utility for converting JSONPath expressions to Polars expressions for efficient querying of JSON data in Polars DataFrames.

## Overview

Polar Express provides a bridge between JSONPath query syntax and Polars expressions, allowing you to:
- Query JSON data stored in Polars DataFrames using familiar JSONPath syntax
- Extract data from nested JSON structures
- Filter arrays using predicates
- Access specific array elements by index

## Installation

```bash
pip install polars
# Clone this repository
git clone https://github.com/evanochsner/polar-express.git
cd polar-express
```

## Usage

```python
import polars as pl
from jsonpath_to_polars import jsonpath_to_polars

# Create a DataFrame with JSON data in string columns
df = pl.DataFrame({
    "user_data": ['{"name": "Alice", "address": {"city": "New York"}}']
})

# Extract data using JSONPath
expr = jsonpath_to_polars("$.user_data.address.city")
result = df.with_columns([
    expr.alias("city")
])

print(result)
```

## Supported JSONPath Features

- Simple field access: `$.field`
- Nested field access: `$.parent.child`
- Array index access: `$.array[0]`
- Nested array-object access: `$.users[0].name`
- Multiple array indices: `$.matrix[0][1]`
- Array filtering with predicates: `$.items[?(@.price>100)].name`

## Development

```bash
# Install development dependencies
pip install pytest pytest-xdist mypy black ruff

# Run tests
./run_tests.py

# Run type checking
mypy jsonpath_to_polars.py

# Format code
black . --line-length 120

# Lint code
ruff .
```

## License

MIT