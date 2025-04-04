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
pip install polars<1.10.0
# Clone this repository
git clone https://github.com/evanochsner/polar-express.git
cd polar-express
pip install -e .
```

## Usage

```python
import polars as pl
from polar_express import jsonpath_to_polars

# Create a DataFrame with JSON data in string columns
df = pl.DataFrame({"user_data": ['{"name": "Alice", "address": {"city": "New York"}}']})

# Extract data using JSONPath
expr = jsonpath_to_polars("$.user_data.address.city")
result = df.with_columns([expr.alias("city")])

print(result)
```

For more examples, check the `examples/` directory:
- `basic_usage.py`: Simple extraction of fields from JSON
- `advanced_usage.py`: Building ETL pipelines with PolarMapper

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
mypy polar_express/

# Format code
black . --line-length 120

# Lint code
ruff .
```

## Package Structure

```
polar_express/
├── polar_express/            # Main package directory
│   ├── __init__.py           # Package exports
│   ├── core/                 # Core functionality
│   │   ├── jsonpath_expr.py  # JSONPathExpr class
│   │   ├── polar_mapper.py   # PolarMapper class 
│   ├── parsing/              # Parsing components
│   │   ├── predicate_parser.py  # Logic for parsing predicates
│   │   ├── path_parser.py    # Path parsing logic
│   ├── conversion/           # Conversion logic
│   │   ├── jsonpath_to_polars.py  # Main conversion function
│   │   ├── handlers/         # Specialized handlers
│   │       ├── array_handlers.py
│   │       ├── field_handlers.py
│   ├── utils/                # Utility functions
│       ├── tokens.py         # Token-related utilities
├── examples/                 # Example usage scripts
├── tests/                    # Tests
```

## License

MIT