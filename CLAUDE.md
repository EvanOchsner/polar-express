# Polar Express - JSONPath to Polars Converter

## Development Commands
- **Install**: `pip install polars pytest pytest-xdist mypy black ruff`
- **Run Tests**:
  - All unit tests: `./run_tests.py`
  - Specific test: `./run_tests.py -t test_function_name`
  - Integration tests: `./run_tests.py --integration`
  - Verbose output: `./run_tests.py -v`
- **Run directly with pytest**:
  - All tests: `python -m pytest`
  - Single test: `python -m pytest tests/test_file.py::TestClass::test_function -v`
- **Typecheck**: `mypy jsonpath_to_polars.py`
- **Format**: `black . --line-length 120`
- **Lint**: `ruff .`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local. Use absolute imports.
- **Typing**: Use type hints for all functions and variables.
- **Docstrings**: Use Google-style docstrings with Args, Returns, and Raises sections.
- **Naming**: 
  - snake_case for variables/functions
  - CamelCase for classes
  - Use descriptive names
- **Error Handling**: Use specific exception types with descriptive messages.
- **Line Length**: 120 characters max
- **JSONPath Syntax**: Always use parentheses with predicates, e.g., `?(@.field>0)`