"""
Example usage of the TableMapper class in Polar Express.

This example demonstrates how to use TableMapper to define table schemas
and extract structured data from JSON using JSONPath expressions.
"""

import polars as pl
import json
from polar_express.core.table_mapper import TableMapper

# Sample JSON data with user information
json_data = """
[
    {
        "id": 1,
        "name": "Alice Smith",
        "email": "alice@example.com",
        "profile": {
            "age": 28,
            "occupation": "Software Engineer",
            "location": {
                "city": "New York",
                "country": "USA"
            }
        },
        "skills": ["Python", "SQL", "JavaScript"],
        "projects": [
            {"name": "Data Pipeline", "status": "Completed"},
            {"name": "Web Dashboard", "status": "In Progress"}
        ]
    },
    {
        "id": 2,
        "name": "Bob Johnson",
        "email": "bob@example.com",
        "profile": {
            "age": 32,
            "occupation": "Data Scientist",
            "location": {
                "city": "San Francisco",
                "country": "USA"
            }
        },
        "skills": ["Python", "R", "Machine Learning"],
        "projects": [
            {"name": "ML Model", "status": "Completed"},
            {"name": "Data Visualization", "status": "Completed"}
        ]
    }
]
"""

# Parse JSON and create DataFrame
data = json.loads(json_data)
df = pl.DataFrame({"data": [json.dumps(data)]})

# Create a table mapper with basic fields
basic_mapper = TableMapper()
basic_mapper.add_column("user_id", "$.data[*].id")
basic_mapper.add_column("name", "$.data[*].name")
basic_mapper.add_column("email", "$.data[*].email")

# Map the data using the basic mapper
basic_result = basic_mapper.map(df)
print("Basic table with simple fields:")
print(basic_result)
print("\n")

# Add nested fields using method chaining
profile_mapper = (
    TableMapper()
    .add_column("user_id", "$.data[*].id")
    .add_column("name", "$.data[*].name")
    .add_column("age", "$.data[*].profile.age")
    .add_column("occupation", "$.data[*].profile.occupation")
    .add_column("city", "$.data[*].profile.location.city")
    .add_column("country", "$.data[*].profile.location.country")
)

# Map the data using the profile mapper
profile_result = profile_mapper.map(df)
print("Table with nested profile fields:")
print(profile_result)
print("\n")

# Create a mapper from a schema dictionary
skills_schema = {"user_id": "$.data[*].id", "name": "$.data[*].name", "skills": "$.data[*].skills[*]"}
skills_mapper = TableMapper.from_schema_dict(skills_schema)

# Map the data to extract skills (this will explode the skills array)
skills_result = skills_mapper.map(df)
print("Table with exploded skills array:")
print(skills_result)
print("\n")

# Create a mapper for project data with predicates
projects_mapper = TableMapper()
projects_mapper.add_column("user_id", "$.data[*].id")
projects_mapper.add_column("name", "$.data[*].name")
projects_mapper.add_column("project_name", "$.data[*].projects[*].name")
projects_mapper.add_column("project_status", "$.data[*].projects[*].status")
# Add a filtered column for only completed projects
projects_mapper.add_column("completed_project", "$.data[*].projects[?(@.status=='Completed')].name")

# Map the data using the projects mapper
projects_result = projects_mapper.map(df)
print("Table with project data:")
print(projects_result)
print("\n")

# Print the schema of the projects mapper
print("Projects mapper schema:")
print(projects_mapper)
