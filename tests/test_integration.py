import sys
import os
import tempfile
import json

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import polars as pl
from jsonpath_to_polars import jsonpath_to_polars


def create_test_dataframe() -> pl.DataFrame:
    """Create a test DataFrame with JSON-encoded string columns."""
    # Sample data for testing
    users = [
        {
            "id": 1,
            "name": "Alice",
            "age": 28,
            "address": {"city": "New York", "zip": "10001"},
            "tags": ["developer", "python"],
            "accounts": [
                {"type": "checking", "balance": 1000.50},
                {"type": "savings", "balance": 5000.75}
            ]
        },
        {
            "id": 2,
            "name": "Bob",
            "age": 35,
            "address": {"city": "San Francisco", "zip": "94105"},
            "tags": ["manager", "java"],
            "accounts": [
                {"type": "checking", "balance": 2500.25},
                {"type": "investment", "balance": 15000.00}
            ]
        },
        {
            "id": 3,
            "name": "Charlie",
            "age": 42,
            "address": {"city": "Boston", "zip": "02110"},
            "tags": ["developer", "javascript"],
            "accounts": [
                {"type": "savings", "balance": 8000.00}
            ]
        }
    ]
    
    items = [
        {
            "items": [
                {"id": 101, "name": "Laptop", "price": 1200.00, "in_stock": True},
                {"id": 102, "name": "Phone", "price": 800.00, "in_stock": True},
                {"id": 103, "name": "Headphones", "price": 50.00, "in_stock": False}
            ],
            "store": "Electronics Plus"
        },
        {
            "items": [
                {"id": 201, "name": "Book", "price": 15.00, "in_stock": True},
                {"id": 202, "name": "Notebook", "price": 5.00, "in_stock": True}
            ],
            "store": "Book Store"
        },
        {
            "items": [
                {"id": 301, "name": "Monitor", "price": 300.00, "in_stock": False},
                {"id": 302, "name": "Keyboard", "price": 80.00, "in_stock": True},
                {"id": 303, "name": "Mouse", "price": 25.00, "in_stock": True}
            ],
            "store": "Computer Shop"
        }
    ]
    
    # Convert to JSON strings
    users_json = [json.dumps(user) for user in users]
    items_json = [json.dumps(item) for item in items]
    
    # Create DataFrame
    return pl.DataFrame({
        "user_data": users_json,
        "inventory": items_json
    })


@pytest.mark.integration
class TestJsonPathIntegration:
    """Test JSONPath to Polars expressions with actual data extraction."""
    
    @pytest.fixture
    def sample_df(self):
        """Fixture to create a sample DataFrame."""
        return create_test_dataframe()
    
    @pytest.fixture
    def parquet_file(self, sample_df):
        """Create a temporary Parquet file with sample data."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        
        sample_df.write_parquet(tmp_path)
        yield tmp_path
        os.unlink(tmp_path)

    def test_simple_field_extraction(self, sample_df):
        """Test extracting a simple field from JSON."""
        # Extract user name
        expr = jsonpath_to_polars("$.user_data.name")
        result = sample_df.with_columns([
            expr.alias("name")
        ])
        
        # Check the extracted data
        assert result.select("name").to_series().to_list() == ["Alice", "Bob", "Charlie"]
    
    def test_nested_field_extraction(self, sample_df):
        """Test extracting a nested field from JSON."""
        # Extract city from address
        expr = jsonpath_to_polars("$.user_data.address.city")
        result = sample_df.with_columns([
            expr.alias("city")
        ])
        
        # Check the extracted data
        assert result.select("city").to_series().to_list() == ["New York", "San Francisco", "Boston"]
    
    def test_array_element_extraction(self, sample_df):
        """Test extracting an element from a JSON array."""
        # Extract first tag
        expr = jsonpath_to_polars("$.user_data.tags[0]")
        result = sample_df.with_columns([
            expr.alias("first_tag")
        ])
        
        # Check the extracted data
        assert result.select("first_tag").to_series().to_list() == ["developer", "manager", "developer"]
    
    def test_array_nested_object_extraction(self, sample_df):
        """Test extracting data from objects inside arrays."""
        # Extract balance from first account
        expr = jsonpath_to_polars("$.user_data.accounts[0].balance")
        result = sample_df.with_columns([
            expr.alias("first_account_balance")
        ])
        
        # Check the extracted data (values are returned as strings from JSON)
        balances = result.select("first_account_balance").to_series().to_list()
        assert len(balances) == 3
        # Convert to float for comparison
        assert abs(float(balances[0]) - 1000.50) < 0.001
        assert abs(float(balances[1]) - 2500.25) < 0.001
        assert abs(float(balances[2]) - 8000.00) < 0.001
    
    def test_array_filter_with_predicate(self, sample_df):
        """Test filtering arrays with predicates."""
        # Extract items with price > 100
        expr = jsonpath_to_polars("$.inventory.items[?(@.price>100)].name")
        
        # The implementation might return a list of matching names or a single name
        # Adjust the assertion based on your implementation
        result = sample_df.with_columns([
            expr.alias("expensive_items")
        ])
        
        # Check the extracted data (implementation specific)
        # This test's assertion may need adjustments based on how predicate filtering is implemented
        assert "Laptop" in str(result.select("expensive_items"))
        assert "Monitor" in str(result.select("expensive_items"))
    
    def test_from_parquet_file(self, parquet_file):
        """Test loading and processing data from a Parquet file."""
        # Read the Parquet file
        df = pl.read_parquet(parquet_file)
        
        # Extract user ages
        expr = jsonpath_to_polars("$.user_data.age")
        result = df.with_columns([
            expr.alias("age")
        ])
        
        # Check the extracted data - values are returned as strings from JSON
        assert result.select("age").to_series().to_list() == ['28', '35', '42']
        
    def test_multiple_extractions(self, sample_df):
        """Test extracting multiple fields in one go."""
        # Extract multiple fields
        name_expr = jsonpath_to_polars("$.user_data.name")
        city_expr = jsonpath_to_polars("$.user_data.address.city")
        
        result = sample_df.with_columns([
            name_expr.alias("name"),
            city_expr.alias("city")
        ])
        
        # Check the extracted data
        assert result.select("name").to_series().to_list() == ["Alice", "Bob", "Charlie"]
        assert result.select("city").to_series().to_list() == ["New York", "San Francisco", "Boston"]