"""Tests for token utilities in polar_express.utils.tokens."""

import pytest
from polar_express.utils.tokens import tokenize_path, handle_predicate_token


class TestTokenizePath:
    """Test tokenize_path function with a wide variety of JSONPath patterns."""

    def test_simple_field_access(self):
        """Test simple field access like 'foo'."""
        path = "foo"
        tokens = tokenize_path(path)
        assert tokens == [("field", "foo")]

    def test_nested_field_access(self):
        """Test nested field access like 'foo.bar.baz'."""
        path = "foo.bar.baz"
        tokens = tokenize_path(path)
        assert tokens == [("field", "foo"), ("field", "bar"), ("field", "baz")]

    def test_field_with_underscore(self):
        """Test field with underscore like 'user_data'."""
        path = "user_data"
        tokens = tokenize_path(path)
        assert tokens == [("field", "user_data")]

    def test_field_with_numeric_characters(self):
        """Test field with numeric characters like 'item123'."""
        path = "item123"
        tokens = tokenize_path(path)
        assert tokens == [("field", "item123")]

    def test_array_index_access(self):
        """Test array index access like 'items[0]'."""
        path = "items[0]"
        tokens = tokenize_path(path)
        assert tokens == [("field", "items"), ("index", 0)]

    def test_array_negative_index(self):
        """Test array negative index like 'items[-1]'."""
        path = "items[-1]"
        tokens = tokenize_path(path)
        assert tokens == [("field", "items"), ("index", -1)]

    def test_nested_array_indices(self):
        """Test nested array indices like 'matrix[0][1]'."""
        path = "matrix[0][1]"
        tokens = tokenize_path(path)
        assert tokens == [("field", "matrix"), ("index", 0), ("index", 1)]

    def test_array_index_with_field(self):
        """Test array index with field access like 'users[0].name'."""
        path = "users[0].name"
        tokens = tokenize_path(path)
        assert tokens == [("field", "users"), ("index", 0), ("field", "name")]

    def test_wildcard_array(self):
        """Test wildcard array access like 'users[*]'."""
        path = "users[*]"
        tokens = tokenize_path(path)
        assert tokens == [("field", "users"), ("wildcard", None)]

    def test_wildcard_with_field(self):
        """Test wildcard with field access like 'users[*].name'."""
        path = "users[*].name"
        tokens = tokenize_path(path)
        assert tokens == [("field", "users"), ("wildcard", None), ("field", "name")]

    def test_multiple_wildcards(self):
        """Test multiple wildcards like 'departments[*].employees[*].name'."""
        path = "departments[*].employees[*].name"
        tokens = tokenize_path(path)
        assert tokens == [
            ("field", "departments"), 
            ("wildcard", None), 
            ("field", "employees"), 
            ("wildcard", None), 
            ("field", "name")
        ]

    def test_simple_predicate(self):
        """Test simple predicate filter like 'users[?(@.age>30)]'."""
        path = "users[?(@.age>30)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert "expr" in tokens[1][1]
        assert "fields" in tokens[1][1]
        assert tokens[1][1]["fields"] == ["age"]

    def test_predicate_with_field(self):
        """Test predicate with field access like 'users[?(@.age>30)].name'."""
        path = "users[?(@.age>30)].name"
        tokens = tokenize_path(path)
        assert len(tokens) == 3
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[2] == ("field", "name")
        assert "expr" in tokens[1][1]
        assert "fields" in tokens[1][1]
        assert tokens[1][1]["fields"] == ["age"]

    def test_compound_predicate_with_and(self):
        """Test compound predicate with AND like 'users[?(@.age>30 && @.active==true)]'."""
        path = "users[?(@.age>30 && @.active==true)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert "expr" in tokens[1][1]
        assert "fields" in tokens[1][1]
        assert set(tokens[1][1]["fields"]) == {"age", "active"}

    def test_compound_predicate_with_or(self):
        """Test compound predicate with OR like 'users[?(@.age>30 || @.premium==true)]'."""
        path = "users[?(@.age>30 || @.premium==true)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert "expr" in tokens[1][1]
        assert "fields" in tokens[1][1]
        assert set(tokens[1][1]["fields"]) == {"age", "premium"}

    def test_complex_predicate_with_nested_conditions(self):
        """Test complex predicate with nested conditions like 'users[?(@.age>30 && (@.premium==true || @.subscribed>6))]'."""
        path = "users[?(@.age>30 && (@.premium==true || @.subscribed>6))]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert "expr" in tokens[1][1]
        assert "fields" in tokens[1][1]
        assert set(tokens[1][1]["fields"]) == {"age", "premium", "subscribed"}

    def test_array_slices(self):
        """Test array slices like 'items[1:3]'."""
        path = "items[1:3]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "items")
        assert tokens[1][0] == "index_expr"
        assert tokens[1][1] == "1:3"

    def test_nested_array_with_predicate(self):
        """Test nested array with predicate like 'departments[0].employees[?(@.position=="manager")]'."""
        path = "departments[0].employees[?(@.position==\"manager\")]"
        tokens = tokenize_path(path)
        assert len(tokens) == 4
        assert tokens[0] == ("field", "departments")
        assert tokens[1] == ("index", 0)
        assert tokens[2] == ("field", "employees")
        assert tokens[3][0] == "predicate"
        assert "expr" in tokens[3][1]
        assert "fields" in tokens[3][1]
        assert tokens[3][1]["fields"] == ["position"]

    def test_complex_path_with_multiple_types(self):
        """Test complex path with multiple types like 'users[0].addresses[*].city'."""
        path = "users[0].addresses[*].city"
        tokens = tokenize_path(path)
        assert tokens == [
            ("field", "users"),
            ("index", 0),
            ("field", "addresses"),
            ("wildcard", None),
            ("field", "city")
        ]

    def test_predicate_with_string_comparison(self):
        """Test predicate with string comparison like 'users[?(@.name=="John")]'."""
        path = 'users[?(@.name=="John")]'
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert "expr" in tokens[1][1]
        assert "fields" in tokens[1][1]
        assert tokens[1][1]["fields"] == ["name"]

    def test_array_index_after_predicate(self):
        """Test array index after predicate like 'users[?(@.age>30)][0]'."""
        path = "users[?(@.age>30)][0]"
        tokens = tokenize_path(path)
        assert len(tokens) == 3
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[2] == ("index", 0)

    def test_complex_nested_path(self):
        """Test complex nested path like 'schools[0].classes[*].students[?(@.grade=="A")].name'."""
        path = 'schools[0].classes[*].students[?(@.grade=="A")].name'
        tokens = tokenize_path(path)
        assert len(tokens) == 7
        assert tokens[0] == ("field", "schools")
        assert tokens[1] == ("index", 0)
        assert tokens[2] == ("field", "classes")
        assert tokens[3] == ("wildcard", None)
        assert tokens[4] == ("field", "students")
        assert tokens[5][0] == "predicate"
        assert tokens[6] == ("field", "name")
        assert tokens[5][1]["fields"] == ["grade"]

    def test_very_deep_nesting(self):
        """Test very deep nesting like 'a.b.c.d.e.f.g.h'."""
        path = "a.b.c.d.e.f.g.h"
        tokens = tokenize_path(path)
        assert tokens == [
            ("field", "a"),
            ("field", "b"),
            ("field", "c"),
            ("field", "d"),
            ("field", "e"),
            ("field", "f"),
            ("field", "g"),
            ("field", "h")
        ]

    def test_multiple_array_indices_with_deep_nesting(self):
        """Test multiple array indices with deep nesting like 'a[0].b[1].c[2].d[3]'."""
        path = "a[0].b[1].c[2].d[3]"
        tokens = tokenize_path(path)
        assert tokens == [
            ("field", "a"),
            ("index", 0),
            ("field", "b"),
            ("index", 1),
            ("field", "c"),
            ("index", 2),
            ("field", "d"),
            ("index", 3)
        ]

    def test_predicate_with_less_than_operator(self):
        """Test predicate with less than operator like 'items[?(@.price<100)]'."""
        path = "items[?(@.price<100)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "items")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1]["fields"] == ["price"]

    def test_predicate_with_greater_than_equal_operator(self):
        """Test predicate with greater than equal operator like 'items[?(@.rating>=4.5)]'."""
        path = "items[?(@.rating>=4.5)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "items")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1]["fields"] == ["rating"]

    def test_predicate_with_boolean_value(self):
        """Test predicate with boolean value like 'features[?(@.enabled==true)]'."""
        path = "features[?(@.enabled==true)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "features")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1]["fields"] == ["enabled"]

    def test_non_standard_index_expressions(self):
        """Test non-standard index expressions that should be captured as index_expr."""
        expressions = ["start:end", "::step", "start:end:step", "start:", ":end"]
        for expr in expressions:
            path = f"items[{expr}]"
            tokens = tokenize_path(path)
            assert len(tokens) == 2
            assert tokens[0] == ("field", "items")
            assert tokens[1][0] == "index_expr"
            assert tokens[1][1] == expr


class TestHandlePredicateToken:
    """Test handle_predicate_token function with various predicate expressions."""

    def test_simple_equality_predicate(self):
        """Test handling a simple equality predicate."""
        path = 'users[?(@.name=="John")]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert token[1]["fields"] == ["name"]

    def test_numeric_comparison_predicate(self):
        """Test handling a numeric comparison predicate."""
        path = 'items[?(@.price>100)]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert token[1]["fields"] == ["price"]

    def test_boolean_predicate(self):
        """Test handling a boolean predicate."""
        path = 'features[?(@.enabled==true)]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert token[1]["fields"] == ["enabled"]

    def test_compound_and_predicate(self):
        """Test handling a compound AND predicate."""
        path = 'users[?(@.age>30 && @.active==true)]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert set(token[1]["fields"]) == {"age", "active"}

    def test_compound_or_predicate(self):
        """Test handling a compound OR predicate."""
        path = 'products[?(@.price<10 || @.featured==true)]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert set(token[1]["fields"]) == {"price", "featured"}

    def test_complex_nested_predicate(self):
        """Test handling a complex nested predicate."""
        path = 'data[?(@.count>100 && (@.status=="active" || @.priority>3))]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert set(token[1]["fields"]) == {"count", "status", "priority"}

    def test_less_than_equal_operator(self):
        """Test handling a less than or equal operator."""
        path = 'scores[?(@.value<=50)]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert token[1]["fields"] == ["value"]

    def test_not_equal_operator(self):
        """Test handling a not equal operator."""
        path = 'users[?(@.status!="inactive")]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert token[1]["fields"] == ["status"]

    def test_multiple_field_references(self):
        """Test handling a predicate with multiple field references."""
        path = 'orders[?(@.total>100 && @.status=="shipped" && @.priority>2)]'
        start_idx = path.index('?')
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert "expr" in token[1]
        assert "fields" in token[1]
        assert set(token[1]["fields"]) == {"total", "status", "priority"}