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
                {"type": "savings", "balance": 5000.75},
            ],
        },
        {
            "id": 2,
            "name": "Bob",
            "age": 35,
            "address": {"city": "San Francisco", "zip": "94105"},
            "tags": ["manager", "java"],
            "accounts": [
                {"type": "checking", "balance": 2500.25},
                {"type": "investment", "balance": 15000.00},
            ],
        },
        {
            "id": 3,
            "name": "Charlie",
            "age": 42,
            "address": {"city": "Boston", "zip": "02110"},
            "tags": ["developer", "javascript"],
            "accounts": [{"type": "savings", "balance": 8000.00}],
        },
    ]

    items = [
        {
            "items": [
                {"id": 101, "name": "Laptop", "price": 1200.00, "in_stock": True},
                {"id": 102, "name": "Phone", "price": 800.00, "in_stock": True},
                {"id": 103, "name": "Headphones", "price": 50.00, "in_stock": False},
            ],
            "store": "Electronics Plus",
        },
        {
            "items": [
                {"id": 201, "name": "Book", "price": 15.00, "in_stock": True},
                {"id": 202, "name": "Notebook", "price": 5.00, "in_stock": True},
            ],
            "store": "Book Store",
        },
        {
            "items": [
                {"id": 301, "name": "Monitor", "price": 300.00, "in_stock": False},
                {"id": 302, "name": "Keyboard", "price": 80.00, "in_stock": True},
                {"id": 303, "name": "Mouse", "price": 25.00, "in_stock": True},
            ],
            "store": "Computer Shop",
        },
    ]

    # Convert to JSON strings
    users_json = [json.dumps(user) for user in users]
    items_json = [json.dumps(item) for item in items]

    # Create DataFrame
    return pl.DataFrame({"user_data": users_json, "inventory": items_json})


# @pytest.mark.integration
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
        result = sample_df.with_columns([expr.alias("name")])

        # Check the extracted data
        assert result.select("name").to_series().to_list() == [
            "Alice",
            "Bob",
            "Charlie",
        ]

    def test_nested_field_extraction(self, sample_df):
        """Test extracting a nested field from JSON."""
        # Extract city from address
        expr = jsonpath_to_polars("$.user_data.address.city")
        result = sample_df.with_columns([expr.alias("city")])

        # Check the extracted data
        assert result.select("city").to_series().to_list() == [
            "New York",
            "San Francisco",
            "Boston",
        ]

    def test_array_element_extraction(self, sample_df):
        """Test extracting an element from a JSON array."""
        # Extract first tag
        expr = jsonpath_to_polars("$.user_data.tags[0]")
        result = sample_df.with_columns([expr.alias("first_tag")])

        # Check the extracted data
        assert result.select("first_tag").to_series().to_list() == [
            "developer",
            "manager",
            "developer",
        ]

    def test_array_nested_object_extraction(self, sample_df):
        """Test extracting data from objects inside arrays."""
        # Extract balance from first account
        expr = jsonpath_to_polars("$.user_data.accounts[0].balance")
        result = sample_df.with_columns([expr.alias("first_account_balance")])

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
        result = sample_df.with_columns([expr.alias("expensive_items")])

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
        result = df.with_columns([expr.alias("age")])

        # Check the extracted data - values are returned as strings from JSON
        assert result.select("age").to_series().to_list() == ["28", "35", "42"]

    def test_multiple_extractions(self, sample_df):
        """Test extracting multiple fields in one go."""
        # Extract multiple fields
        name_expr = jsonpath_to_polars("$.user_data.name")
        city_expr = jsonpath_to_polars("$.user_data.address.city")

        result = sample_df.with_columns(
            [name_expr.alias("name"), city_expr.alias("city")]
        )

        # Check the extracted data
        assert result.select("name").to_series().to_list() == [
            "Alice",
            "Bob",
            "Charlie",
        ]
        assert result.select("city").to_series().to_list() == [
            "New York",
            "San Francisco",
            "Boston",
        ]

    def test_array_wildcard_access(self, sample_df):
        """Test array wildcard access with actual data."""
        # Extract all tags
        expr = jsonpath_to_polars("$.user_data.tags[*]")
        result = sample_df.with_columns([expr.alias("all_tags")])

        # The results should contain all tags for each user
        all_tags = result.select("all_tags").to_series().to_list()
        assert len(all_tags) == 3
        assert "developer" in str(all_tags[0]) and "python" in str(all_tags[0])
        assert "manager" in str(all_tags[1]) and "java" in str(all_tags[1])
        assert "developer" in str(all_tags[2]) and "javascript" in str(all_tags[2])

    def test_array_negative_index(self, sample_df):
        """Test array negative index access."""
        # Extract the last tag for each user
        expr = jsonpath_to_polars("$.user_data.tags[-1]")
        result = sample_df.with_columns([expr.alias("last_tag")])

        # Check the extracted data
        assert result.select("last_tag").to_series().to_list() == [
            "python",
            "java",
            "javascript",
        ]

    def test_multiple_array_indices(self, sample_df):
        """Test accessing nested arrays with specific indices."""
        # Create a temporary column with nested arrays for testing
        json_with_nested_arrays = [
            json.dumps({"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}),
            json.dumps({"matrix": [[10, 11, 12], [13, 14, 15], [16, 17, 18]]}),
            json.dumps({"matrix": [[19, 20, 21], [22, 23, 24], [25, 26, 27]]}),
        ]

        # Create a new DataFrame since we need to access directly without a parent column
        df_with_matrices = pl.DataFrame({"matrix_data": json_with_nested_arrays})

        # Extract element at position [0][1] (second element of first array)
        expr = jsonpath_to_polars("$.matrix_data.matrix[0][1]")
        result = df_with_matrices.with_columns([expr.alias("matrix_element")])

        # Check the extracted data
        assert result.select("matrix_element").to_series().to_list() == [
            "2",
            "11",
            "20",
        ]

    def test_wildcard_with_deeply_nested_field(self, sample_df):
        """Test wildcard with deeply nested fields."""
        # Create a temporary column with deeply nested data
        json_with_deep_nesting = [
            json.dumps(
                {
                    "users": [
                        {"contact": {"address": {"city": "Seattle", "state": "WA"}}},
                        {"contact": {"address": {"city": "Portland", "state": "OR"}}},
                    ]
                }
            ),
            json.dumps(
                {
                    "users": [
                        {"contact": {"address": {"city": "Chicago", "state": "IL"}}},
                        {"contact": {"address": {"city": "Detroit", "state": "MI"}}},
                    ]
                }
            ),
            json.dumps(
                {"users": [{"contact": {"address": {"city": "Austin", "state": "TX"}}}]}
            ),
        ]

        # Create a new DataFrame
        df_with_nested = pl.DataFrame({"nested_data": json_with_deep_nesting})

        # Extract cities from all users
        expr = jsonpath_to_polars("$.nested_data.users[*].contact.address.city")
        result = df_with_nested.with_columns([expr.alias("cities")])

        # Check that the extracted data contains the expected cities
        cities = result.select("cities").to_series().to_list()
        assert len(cities) == 3
        assert "Seattle" in str(cities[0]) and "Portland" in str(cities[0])
        assert "Chicago" in str(cities[1]) and "Detroit" in str(cities[1])
        assert "Austin" in str(cities[2])

    def test_multiple_arrays_with_wildcards(self, sample_df):
        """Test path with multiple array wildcards."""
        # Create data with nested arrays
        json_with_multiple_arrays = [
            json.dumps(
                {
                    "departments": [
                        {
                            "name": "Engineering",
                            "employees": [
                                {"name": "John", "position": "Developer"},
                                {"name": "Lisa", "position": "Manager"},
                            ],
                        },
                        {
                            "name": "Marketing",
                            "employees": [{"name": "Steve", "position": "Director"}],
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "departments": [
                        {
                            "name": "Sales",
                            "employees": [
                                {"name": "Mark", "position": "Account Executive"},
                                {"name": "Sarah", "position": "Sales Rep"},
                            ],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "departments": [
                        {
                            "name": "HR",
                            "employees": [{"name": "Emily", "position": "HR Manager"}],
                        },
                        {
                            "name": "Finance",
                            "employees": [
                                {"name": "Tom", "position": "Accountant"},
                                {"name": "Jessica", "position": "Financial Analyst"},
                            ],
                        },
                    ]
                }
            ),
        ]

        # Create a new DataFrame
        df_with_departments = pl.DataFrame({"org_data": json_with_multiple_arrays})

        # Extract all employee names across all departments
        expr = jsonpath_to_polars("$.org_data.departments[*].employees[*].name")
        result = df_with_departments.with_columns([expr.alias("employee_names")])

        # Check that the extracted data contains the expected names
        names = result.select("employee_names").to_series().to_list()
        assert len(names) == 3
        assert (
            "John" in str(names[0])
            and "Lisa" in str(names[0])
            and "Steve" in str(names[0])
        )
        assert "Mark" in str(names[1]) and "Sarah" in str(names[1])
        assert (
            "Emily" in str(names[2])
            and "Tom" in str(names[2])
            and "Jessica" in str(names[2])
        )

    def test_array_index_with_nested_arrays(self, sample_df):
        """Test array index access with nested arrays."""
        # Create data with specific array indices and nested arrays
        json_with_schools = [
            json.dumps(
                {
                    "schools": [
                        {
                            "name": "High School",
                            "classes": [
                                {
                                    "name": "Math",
                                    "students": [
                                        {"name": "Alex", "grade": "A"},
                                        {"name": "Brianna", "grade": "B"},
                                    ],
                                },
                                {
                                    "name": "Science",
                                    "students": [
                                        {"name": "Charlie", "grade": "A-"},
                                        {"name": "Dana", "grade": "B+"},
                                    ],
                                },
                            ],
                        },
                        {
                            "name": "Middle School",
                            "classes": [
                                {
                                    "name": "History",
                                    "students": [
                                        {"name": "Evan", "grade": "A"},
                                        {"name": "Fiona", "grade": "A-"},
                                    ],
                                }
                            ],
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "schools": [
                        {
                            "name": "Elementary",
                            "classes": [
                                {
                                    "name": "Reading",
                                    "students": [
                                        {"name": "Greg", "grade": "A+"},
                                        {"name": "Hannah", "grade": "A"},
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "schools": [
                        {
                            "name": "College",
                            "classes": [
                                {
                                    "name": "Computer Science",
                                    "students": [
                                        {"name": "Ian", "grade": "B+"},
                                        {"name": "Julia", "grade": "A"},
                                    ],
                                },
                                {
                                    "name": "Economics",
                                    "students": [
                                        {"name": "Kevin", "grade": "B"},
                                        {"name": "Laura", "grade": "A-"},
                                    ],
                                },
                            ],
                        }
                    ]
                }
            ),
        ]

        # Create a new DataFrame
        df_with_schools = pl.DataFrame({"education_data": json_with_schools})

        # Extract all grades from all students in the first school's classes
        expr = jsonpath_to_polars(
            "$.education_data.schools[0].classes[*].students[*].grade"
        )
        result = df_with_schools.with_columns(
            [expr.alias("grades")]
        )  # Check that the extracted data contains the expected grades
        grades = result.select("grades").to_series().to_list()
        assert len(grades) == 3

        # First row should contain grades from High School (first school)
        assert (
            "A" in str(grades[0])
            and "B" in str(grades[0])
            and "A-" in str(grades[0])
            and "B+" in str(grades[0])
        )

        # Second row should contain grades from Elementary School (first school)
        assert "A+" in str(grades[1]) and "A" in str(grades[1])

        # Third row should contain grades from College (first school)
        assert (
            "B+" in str(grades[2])
            and "A" in str(grades[2])
            and "B" in str(grades[2])
            and "A-" in str(grades[2])
        )
