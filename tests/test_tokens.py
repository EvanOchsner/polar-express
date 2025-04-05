"""Tests for token utilities in polar_express.utils.tokens."""

import pytest

from polar_express.utils.tokens import handle_predicate_token, tokenize_path, tokens_to_jsonpath


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
            ("field", "name"),
        ]

    def test_simple_predicate(self):
        """Test simple predicate filter like 'users[?(@.age>30)]'."""
        path = "users[?(@.age>30)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == "@.age>30"

    def test_predicate_with_field(self):
        """Test predicate with field access like 'users[?(@.age>30)].name'."""
        path = "users[?(@.age>30)].name"
        tokens = tokenize_path(path)
        assert len(tokens) == 3
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[2] == ("field", "name")
        assert tokens[1][1] == "@.age>30"

    def test_compound_predicate_with_and(self):
        """Test compound predicate with AND like 'users[?(@.age>30 && @.active==true)]'."""
        path = "users[?(@.age>30 && @.active==true)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == "@.age>30 && @.active==true"

    def test_compound_predicate_with_or(self):
        """Test compound predicate with OR like 'users[?(@.age>30 || @.premium==true)]'."""
        path = "users[?(@.age>30 || @.premium==true)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == "@.age>30 || @.premium==true"

    def test_complex_predicate_with_nested_conditions(self):
        """
        Test complex predicate with nested conditions like
        'users[?(@.age>30 && (@.premium==true || @.subscribed>6))]'.
        """
        path = "users[?(@.age>30 && (@.premium==true || @.subscribed>6))]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == "@.age>30 && (@.premium==true || @.subscribed>6)"

    @pytest.mark.xfail(reason="Non-integer array indices are not supported")
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
        path = 'departments[0].employees[?(@.position=="manager")]'
        tokens = tokenize_path(path)
        assert len(tokens) == 4
        assert tokens[0] == ("field", "departments")
        assert tokens[1] == ("index", 0)
        assert tokens[2] == ("field", "employees")
        assert tokens[3][0] == "predicate"
        assert tokens[3][1] == '@.position=="manager"'

    def test_complex_path_with_multiple_types(self):
        """Test complex path with multiple types like 'users[0].addresses[*].city'."""
        path = "users[0].addresses[*].city"
        tokens = tokenize_path(path)
        assert tokens == [
            ("field", "users"),
            ("index", 0),
            ("field", "addresses"),
            ("wildcard", None),
            ("field", "city"),
        ]

    def test_predicate_with_string_comparison(self):
        """Test predicate with string comparison like 'users[?(@.name=="John")]'."""
        path = 'users[?(@.name=="John")]'
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "users")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == '@.name=="John"'

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
        assert tokens[5][1] == '@.grade=="A"'

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
            ("field", "h"),
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
            ("index", 3),
        ]

    def test_predicate_with_less_than_operator(self):
        """Test predicate with less than operator like 'items[?(@.price<100)]'."""
        path = "items[?(@.price<100)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "items")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == "@.price<100"

    def test_predicate_with_greater_than_equal_operator(self):
        """Test predicate with greater than equal operator like 'items[?(@.rating>=4.5)]'."""
        path = "items[?(@.rating>=4.5)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "items")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == "@.rating>=4.5"

    def test_predicate_with_boolean_value(self):
        """Test predicate with boolean value like 'features[?(@.enabled==true)]'."""
        path = "features[?(@.enabled==true)]"
        tokens = tokenize_path(path)
        assert len(tokens) == 2
        assert tokens[0] == ("field", "features")
        assert tokens[1][0] == "predicate"
        assert tokens[1][1] == "@.enabled==true"

    @pytest.mark.xfail(reason="Non-integer array indices are not supported")
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
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == '@.name=="John"'

    def test_numeric_comparison_predicate(self):
        """Test handling a numeric comparison predicate."""
        path = "items[?(@.price>100)]"
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == "@.price>100"

    def test_boolean_predicate(self):
        """Test handling a boolean predicate."""
        path = "features[?(@.enabled==true)]"
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == "@.enabled==true"

    def test_compound_and_predicate(self):
        """Test handling a compound AND predicate."""
        path = "users[?(@.age>30 && @.active==true)]"
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == "@.age>30 && @.active==true"

    def test_compound_or_predicate(self):
        """Test handling a compound OR predicate."""
        path = "products[?(@.price<10 || @.featured==true)]"
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == "@.price<10 || @.featured==true"

    def test_complex_nested_predicate(self):
        """Test handling a complex nested predicate."""
        path = 'data[?(@.count>100 && (@.status=="active" || @.priority>3))]'
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == '@.count>100 && (@.status=="active" || @.priority>3)'

    def test_less_than_equal_operator(self):
        """Test handling a less than or equal operator."""
        path = "scores[?(@.value<=50)]"
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == "@.value<=50"

    def test_not_equal_operator(self):
        """Test handling a not equal operator."""
        path = 'users[?(@.status!="inactive")]'
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == '@.status!="inactive"'

    def test_multiple_field_references(self):
        """Test handling a predicate with multiple field references."""
        path = 'orders[?(@.total>100 && @.status=="shipped" && @.priority>2)]'
        start_idx = path.index("?")
        token = handle_predicate_token(path, start_idx)
        assert token[0] == "predicate"
        assert token[1] == '@.total>100 && @.status=="shipped" && @.priority>2'


class TestTokensToJsonpath:
    """Test tokens_to_jsonpath function for converting tokens back to JSONPath."""

    def test_simple_field_access(self):
        """Test converting tokens for simple field access."""
        tokens = [("field", "foo")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.foo"

    def test_nested_field_access(self):
        """Test converting tokens for nested field access."""
        tokens = [("field", "foo"), ("field", "bar"), ("field", "baz")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.foo.bar.baz"

    def test_field_with_underscore(self):
        """Test converting tokens for field with underscore."""
        tokens = [("field", "user_data")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.user_data"

    def test_field_with_numeric_characters(self):
        """Test converting tokens for field with numeric characters."""
        tokens = [("field", "item123")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.item123"

    def test_array_index_access(self):
        """Test converting tokens for array index access."""
        tokens = [("field", "items"), ("index", 0)]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.items[0]"

    def test_array_negative_index(self):
        """Test converting tokens for array negative index."""
        tokens = [("field", "items"), ("index", -1)]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.items[-1]"

    def test_nested_array_indices(self):
        """Test converting tokens for nested array indices."""
        tokens = [("field", "matrix"), ("index", 0), ("index", 1)]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.matrix[0][1]"

    def test_array_index_with_field(self):
        """Test converting tokens for array index with field access."""
        tokens = [("field", "users"), ("index", 0), ("field", "name")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[0].name"

    def test_wildcard_array(self):
        """Test converting tokens for wildcard array access."""
        tokens = [("field", "users"), ("wildcard", None)]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[*]"

    def test_wildcard_with_field(self):
        """Test converting tokens for wildcard with field access."""
        tokens = [("field", "users"), ("wildcard", None), ("field", "name")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[*].name"

    def test_multiple_wildcards(self):
        """Test converting tokens for multiple wildcards."""
        tokens = [
            ("field", "departments"),
            ("wildcard", None),
            ("field", "employees"),
            ("wildcard", None),
            ("field", "name"),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.departments[*].employees[*].name"

    def test_simple_predicate(self):
        """Test converting tokens for simple predicate filter."""
        tokens = [("field", "users"), ("predicate", "@.age>30")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[?(@.age>30)]"

    def test_predicate_with_field(self):
        """Test converting tokens for predicate with field access."""
        tokens = [("field", "users"), ("predicate", "@.age>30"), ("field", "name")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[?(@.age>30)].name"

    def test_compound_predicate_with_and(self):
        """Test converting tokens for compound predicate with AND."""
        tokens = [
            ("field", "users"),
            ("predicate", "@.age>30 && @.active==true"),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[?(@.age>30 && @.active==true)]"

    def test_compound_predicate_with_or(self):
        """Test converting tokens for compound predicate with OR."""
        tokens = [
            ("field", "users"),
            ("predicate", "@.age>30 || @.premium==true"),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[?(@.age>30 || @.premium==true)]"

    def test_complex_predicate_with_nested_conditions(self):
        """Test converting tokens for complex predicate with nested conditions."""
        tokens = [
            ("field", "users"),
            ("predicate", "@.age>30 && (@.premium==true || @.subscribed>6)"),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[?(@.age>30 && (@.premium==true || @.subscribed>6))]"

    @pytest.mark.xfail(reason="index_expr token type are not supported")
    def test_array_slices(self):
        """Test converting tokens for array slices."""
        tokens = [("field", "items"), ("index_expr", "1:3")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.items[1:3]"

    def test_nested_array_with_predicate(self):
        """Test converting tokens for nested array with predicate."""
        tokens = [
            ("field", "departments"),
            ("index", 0),
            ("field", "employees"),
            ("predicate", '@.position=="manager"'),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == '$.departments[0].employees[?(@.position=="manager")]'

    def test_complex_path_with_multiple_types(self):
        """Test converting tokens for complex path with multiple types."""
        tokens = [("field", "users"), ("index", 0), ("field", "addresses"), ("wildcard", None), ("field", "city")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[0].addresses[*].city"

    def test_predicate_with_string_comparison(self):
        """Test converting tokens for predicate with string comparison."""
        tokens = [("field", "users"), ("predicate", '@.name=="John"')]
        result = tokens_to_jsonpath(tokens)
        assert result == '$.users[?(@.name=="John")]'

    def test_array_index_after_predicate(self):
        """Test converting tokens for array index after predicate."""
        tokens = [("field", "users"), ("predicate", "@.age>30"), ("index", 0)]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.users[?(@.age>30)][0]"

    def test_complex_nested_path(self):
        """Test converting tokens for complex nested path."""
        tokens = [
            ("field", "schools"),
            ("index", 0),
            ("field", "classes"),
            ("wildcard", None),
            ("field", "students"),
            ("predicate", '@.grade=="A"'),
            ("field", "name"),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == '$.schools[0].classes[*].students[?(@.grade=="A")].name'

    def test_very_deep_nesting(self):
        """Test converting tokens for very deep nesting."""
        tokens = [
            ("field", "a"),
            ("field", "b"),
            ("field", "c"),
            ("field", "d"),
            ("field", "e"),
            ("field", "f"),
            ("field", "g"),
            ("field", "h"),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.a.b.c.d.e.f.g.h"

    def test_multiple_array_indices_with_deep_nesting(self):
        """Test converting tokens for multiple array indices with deep nesting."""
        tokens = [
            ("field", "a"),
            ("index", 0),
            ("field", "b"),
            ("index", 1),
            ("field", "c"),
            ("index", 2),
            ("field", "d"),
            ("index", 3),
        ]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.a[0].b[1].c[2].d[3]"

    def test_predicate_with_less_than_operator(self):
        """Test converting tokens for predicate with less than operator."""
        tokens = [("field", "items"), ("predicate", "@.price<100")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.items[?(@.price<100)]"

    def test_predicate_with_greater_than_equal_operator(self):
        """Test converting tokens for predicate with greater than equal operator."""
        tokens = [("field", "items"), ("predicate", "@.rating>=4.5")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.items[?(@.rating>=4.5)]"

    def test_predicate_with_boolean_value(self):
        """Test converting tokens for predicate with boolean value."""
        tokens = [("field", "features"), ("predicate", "@.enabled==true")]
        result = tokens_to_jsonpath(tokens)
        assert result == "$.features[?(@.enabled==true)]"

    @pytest.mark.xfail(reason="index_expr token type are not supported")
    def test_non_standard_index_expressions(self):
        """Test converting tokens for non-standard index expressions."""
        expressions = ["start:end", "::step", "start:end:step", "start:", ":end"]
        for expr in expressions:
            tokens = [("field", "items"), ("index_expr", expr)]
            result = tokens_to_jsonpath(tokens)
            assert result == f"$.items[{expr}]"

    def test_simple_roundtrip_conversion(self):
        """Test roundtrip conversion for simple paths that don't include predicates."""
        paths = ["foo.bar.baz", "items[0]", "users[*].name", "departments[*].employees[*]"]

        for original_path in paths:
            tokens = tokenize_path(original_path)
            result = tokens_to_jsonpath(tokens)
            # Remove the dollar sign to compare with original (tokenize_path strips the $ prefix)
            assert result[2:] == original_path  # Skip the "$." prefix

    def test_predicate_structure_preserved(self):
        """
        Test that the predicate structure is preserved in roundtrip conversion.

        Note: The exact predicate expression text might be different after conversion to Polars
        expressions, so we only verify that key structural elements are preserved.
        """
        path = "users[?(@.age>30)].name"
        tokens = tokenize_path(path)
        result = tokens_to_jsonpath(tokens)

        # Verify the result matches the expected format
        assert result == "$.users[?(@.age>30)].name"

        # More complex path with nested predicates
        path = 'schools[0].classes[*].students[?(@.grade=="A")].name'
        tokens = tokenize_path(path)
        result = tokens_to_jsonpath(tokens)

        # Verify the result matches the expected format
        assert result == '$.schools[0].classes[*].students[?(@.grade=="A")].name'
